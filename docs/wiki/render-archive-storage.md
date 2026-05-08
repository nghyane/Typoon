# Render archive storage & CDN

How the rendered chapter archive (`render.bnl`) reaches a browser, and
how pipeline artifacts (`prepared.bnl`, `masks.npz`) move between
workers across hosts. Read this if you change archive storage, add a
backend, debug a CDN issue, or set up a multi-host deployment.

## Two storage tiers

Typoon separates storage by audience:

| Tier | Type | Visible to | Holds |
|---|---|---|---|
| `pipeline` | `BlobStore` (no URL) | Workers only | prepared.bnl, masks.npz |
| `public`   | `ArtifactStore` (has URL) | Browser via CDN | render.bnl |

`ArtifactStore` extends `BlobStore` with `url(locator, *, version)` for
browser fetch. Pipeline blobs never get a public URL — they ship
between workers over the trusted plane.

The two tiers are configured independently:

```toml
[storage.public]
type = "huggingface"     # or "local" for dev
hf_repo = "nghyane/mcz-cdn"
cdn_prefix = "https://cdn.bunle.cloud/t"

[storage.pipeline]
type = "http"            # or "local" for dev
http_base_url = "http://100.72.203.52:8000"   # tailnet
# http_api_token via env: TYPOON_PIPELINE_TOKEN
```

## Goals

- Browser reads served by an external CDN, not by the API origin.
- Free egress (Cloudflare/HF have no per-byte charge for what we use).
- Multi-backend ready: same wiring works for Local/HF/Drive/R2/...
- URL build is sync + zero-IO so `list_chapters` cost is O(1) per row.
- Anyone-with-link semantics — no auth on `.bnl` so Range requests are
  cacheable at the edge.
- Locator path is unguessable so the listing isn't enumerable.

## URL shape

```
https://cdn.bunle.cloud/<prefix>/<locator>?v=<updated_at>
```

| Prefix | Origin                                                     | Use |
|--------|------------------------------------------------------------|-----|
| `/t/`  | `huggingface.co/datasets/<HF_REPO>/resolve/main/typoon/`   | Typoon archives — added by us |
| `/h/`  | `huggingface.co/datasets/<rest>`                           | Public HF dataset passthrough |
| `/f/`  | `huggingface.co/buckets/<HF_REPO>/resolve/`                | HF private bucket (Xet protocol upload) |
| `/d/`  | `lh3.googleusercontent.com/d/<file_id>=s0-rw`              | Google Drive |
| `/r/`  | `pub-<uuid>.r2.dev/`                                       | R2 public bucket |
| `/c/`  | `https://<host>/<path>`                                    | Generic passthrough |

The path-after-prefix is the **locator**. Path-based backends (HF, R2,
Local) use a string path. Opaque-id backends (Drive) use the platform's
file id.

`?v=<updated_at>` busts the CDN cache when a chapter re-renders. The
locator path itself is stable per `(project_id, chapter_id)` so warm
cache hits stay hot.

## Locator generation

Path-based backends:

```python
# typoon/adapters/chapter_archive.py
def archive_token(project_id, chapter_id, salt: bytes) -> str:
    msg = f"{project_id}:{chapter_id}".encode()
    digest = hmac.new(salt, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()[:16]

def render_key(p, c, salt) -> str:
    return f"render/{archive_token(p, c, salt)}.bnl"
```

- Deterministic: same `(p, c, salt)` → same locator forever → CDN cache
  key stable across re-renders.
- Unguessable: 96 bits of entropy.
- One-way: locator does not leak project_id / chapter_id.
- Salt rotation = full nuke (every URL changes). Don't routine-rotate.

Opaque-id backends store whatever the platform returned at upload time.

## Code surface

```
typoon/adapters/blob_store.py
  BlobStore Protocol
    backend_name: str (class const)
    put(key, src) -> locator        # async, returns persisted locator
    get(locator, dest)              # async
    delete(locator) -> bool         # async
    exists(locator) -> bool         # async, cheap presence check
    aclose()                        # async, releases pooled clients
  LocalBlobStore                    # filesystem-backed

typoon/adapters/artifact_store.py
  ArtifactStore(BlobStore) Protocol
    url(locator, *, version: str)   # SYNC + zero IO; browser-facing
  LocalArtifactStore                # extends LocalBlobStore + url()
  HuggingFaceArtifactStore          # HF Hub upload, bunle CDN url

typoon/adapters/http_blob_store.py
  HttpBlobStore                     # /api/blobs over HTTP, tailnet-friendly

typoon/adapters/storage_registry.py
  StorageRegistry(pipeline, public, readers)
  build_storage(cfg, paths) -> StorageRegistry

typoon/api/routes/blobs.py
  PUT/GET/HEAD/DELETE /api/blobs/<key>  # require_worker scope

typoon/adapters/chapter_archive.py
  archive_token(p, c, salt) -> 16-char base64url HMAC
  render_key(p, c, salt) -> "render/<token>.bnl"
  pack_and_upload(...) -> (page_count, locator)

typoon/api/deps.py
  get_storage() -> StorageRegistry   # FastAPI dep
  require_worker                     # API token with scope='worker'

typoon/storage/postgres.py
  chapters table: archive_backend TEXT, archive_locator TEXT
  api_tokens table: scopes TEXT[]
  set_archive(chapter_id, backend, locator)

typoon/api/routes/projects.py
  _archive_url(stores, row) — dispatches by row.archive_backend
  ChapterOut.archive_url consumed directly by the FE reader
```

## Coexistence model

Each chapter row carries `(archive_backend, archive_locator)`. The
render worker writes through `stores.public`. The API URL builder
dispatches via `stores.reader(row.archive_backend)`, so chapters
rendered against an old public store keep working after the operator
switches the primary — no migration required.

Pipeline blobs (prepared.bnl, masks.npz) live on `stores.pipeline`
exclusively — they never leak to a browser and the public CDN never
sees them.

## Multi-host topology (production)

Workers, API, storage can all run on different hosts joined into a
single Tailscale tailnet. Pipeline traffic never leaves the mesh; only
the rendered archive crosses to the public CDN.

```
[ Browser ] → [ Cloudflare/bunle CDN ] → [ HuggingFace dataset ]   render.bnl
                                                  ↑
                                                  │ render worker upload
[ User ] → HTTPS → [ API host(s) ] ────────── (tailnet) ──────────┐
                          ↓ Postgres                                │
                     [ DB host ]                                    │
                                                                    │
[ Vision worker(s) ]  [ LLM worker(s) ]  [ Storage host (typoon api │
        │                    │              with TYPOON_API_ROLE=   │
        └─────────tailnet────┴────────── storage) ]    ←──── /api/blobs/...
                                          serves prepared/masks
```

Concrete role wiring:

```bash
# Storage host — serves /api/blobs and stores pipeline blobs on disk
TYPOON_API_ROLE=storage typoon api

# API host — user-facing routes, no blob endpoint
TYPOON_API_ROLE=api typoon api

# Vision worker — scan + render; reads pipeline via tailnet, writes
# render.bnl to the public store
TYPOON_PIPELINE_TYPE=http \
TYPOON_PIPELINE_BASE_URL=http://100.72.203.52:8000 \
TYPOON_PIPELINE_TOKEN=typ_… \
TYPOON_PUBLIC_TYPE=huggingface \
HF_TOKEN=hf_… \
typoon work --role vision

# LLM worker — translate; reads pipeline via tailnet for image context
TYPOON_PIPELINE_TYPE=http \
TYPOON_PIPELINE_BASE_URL=http://100.72.203.52:8000 \
TYPOON_PIPELINE_TOKEN=typ_… \
typoon work --role llm
```

The single-host dev case stays the same (everything `local`), so
nothing in the day-to-day workflow changes for solo development.

## Worker API tokens

`/api/blobs/*` rejects ordinary user JWTs — workers carry an API token
(`typ_…`) issued with `scopes = ['worker']`. Mint one via the existing
token issuance flow with the new `scopes` argument:

```python
from typoon.api.auth_token import issue_api_token
tok_id, plaintext, prefix = await issue_api_token(
    db, user_id=admin_id, name="vision-worker-1", scopes=["worker"],
)
```

The token only authorizes pipeline blob access; it cannot reach
projects/upload/admin endpoints because the scope is checked
explicitly by `require_worker`. Standard user-facing API tokens have
empty scopes and remain unable to hit `/api/blobs/*`.

## Lifecycle of intermediate caches

Three artifacts per chapter, distinct lifetimes:

| File | Visibility | When written | When deleted |
|---|---|---|---|
| `prepared.bnl` | server-only | prepare stage | `typoon prune` (TTL, opt-in) |
| `masks.npz` | server-only | scan stage | `typoon prune` (TTL, opt-in) |
| `render.bnl` | public (browser) | render stage | only on chapter delete or redo |

prepared/masks are **caches**, not state — the pipeline can rebuild
them from the source upload + DB. Keeping them is purely a redo
optimization. After a chapter has been rendered and not touched for a
while, they can be safely freed.

```bash
typoon prune --days 30                # delete cache older than 30d
typoon prune --days 30 --dry-run      # show what would be deleted, don't
```

The render archive is never touched by prune; readers keep working.
Redo after prune still works — it just re-runs the prepare/scan steps
from the source, so it's slower than a cached redo.

Schedule: run weekly or monthly via cron / launchd / systemd timer
depending on host. Or do it manually when disk pressure shows up.

## Bunle CDN

The CDN is a single Cloudflare Pages Function. Source:
`/Users/nghiahoang/Dev/bunle/bunle-cdn/functions/[[path]].ts`.

Behaviour:

- Range request forwarding + `Accept-Ranges: bytes` echo
- 1-year `Cache-Control: public, max-age=31536000, immutable`
- Cache key = full URL including Range header → Range slices share cache
- CORS open for Range / Content-Length / Content-Range
- `cache.put` async via `waitUntil` so first request returns fast

To add a backend later: add one `case` to `resolveOrigin()` returning
the upstream URL, plus a matching prefix in the matching Python adapter's
`url()` method. No other CDN code changes.

### Deploys

`bunle-cdn` exists in two Cloudflare accounts:

1. **Original (production)**: serves `cdn.bunle.cloud`. Owned by the
   account with zone `bunle.cloud`. Credentials not present on this
   machine.
2. **Local copy**: created on the active wrangler account during the
   prototype. Project name `bunle-cdn`, deployment URL
   `bunle-cdn-16g.pages.dev`. The current default of
   `cfg.storage.hf_cdn_prefix` points here so dev works without depending
   on the production CDN.

Rollout to production: add the `/t/` route in the production
`bunle-cdn` repo, deploy, then flip `TYPOON_HF_CDN_PREFIX` to
`https://cdn.bunle.cloud/t`. Old URLs still work — the change is
backwards-compatible.

To redeploy after editing `[[path]].ts`:

```bash
cd /Users/nghiahoang/Dev/bunle/bunle-cdn
wrangler pages deploy . --project-name=bunle-cdn --branch=main --commit-dirty=true
```

The custom domain `cdn.bunle.cloud` was NOT bound from the active
account because zone `bunle.cloud` lives elsewhere. To brand under our
own domain later, bind a zone we do own (e.g. `cdn.mangalocal.com`).

### HF dataset

Repo: `nghyane/mcz-cdn` (also used by bunle-site for upload buckets).
Repo type is `bucket` but it serves files via `/datasets/...resolve/main/...`
just fine. Path layout:

```
nghyane/mcz-cdn/
  ch001.webp                 # bunle-site test fixtures (don't touch)
  ch001-fast.webp
  ch001-api.webp
  test.txt
  typoon/                    # ← Typoon namespace
    render/<token>.bnl
```

Token: `HF_TOKEN` env var. Same token bunle-api uses; loaded from
`~/.cache/huggingface/token` for development.

Upload via `huggingface_hub.upload_file` with `repo_type="dataset"`.
Implementation in `HuggingFaceArtifactStore.put()`.

## Probe checklist

Before assuming a backend is wired up, run:

```bash
# 1. CDN /t/ route reachable for an existing file
curl -sI https://bunle-cdn-16g.pages.dev/t/probe.txt

# 2. Range supported
curl -i -H "Range: bytes=5-15" https://bunle-cdn-16g.pages.dev/t/probe.txt
# expect: HTTP/2 206, Content-Range: bytes 5-15/N

# 3. CORS open
curl -i -X OPTIONS \
  -H "Origin: https://app.example" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: Range" \
  https://bunle-cdn-16g.pages.dev/t/probe.txt

# 4. Edge cache hit on warm fetch
curl -sI https://bunle-cdn-16g.pages.dev/t/probe.txt | grep cf-cache-status
# expect: HIT after first request
```

## Adding a backend (e.g. Google Drive)

1. New file `typoon/adapters/<name>_artifact_store.py` implementing
   `ArtifactStore`. Set `backend_name`. Write `put` to upload + return
   the platform-native id; `url(locator)` returns a stable public URL
   via the matching bunle CDN prefix.
2. Register it in `typoon/api/deps.py:build_artifact_stores()` (one
   `if cfg.storage.<creds>_set: stores[X.backend_name] = X(...)`).
3. Add config fields in `StorageConfig`.
4. Add a `case` in `bunle-cdn/[[path]].ts` if a new prefix is needed,
   redeploy.
5. Done. No DB migration, no stages/api changes.

## Open questions

- **TOS risk on HF**: manga archives might trip HF moderation; have an
  R2 fallback ready before scaling reads.
- **Quota on HF**: per-repo soft limit ~50GB. If we cross, shard repos
  by token prefix (locator first char → routes to typoon-cdn-a/b/c).
- **Drive flow**: bunle-api implements per-user OAuth + share-anyone
  permission. If Typoon adds Drive as a per-user storage option, copy
  that pattern (don't reuse a service refresh-token — quota is per
  account).
- **`cdn.bunle.cloud` ownership**: production custom domain unbound
  from the dev account. Either get credentials for the account that
  owns zone `bunle.cloud` or bind a domain we do own.

## Migration history

- `e216309` added the `archive_backend` / `archive_locator` columns +
  `set_archive` plumbing.
- `df82452` introduced the registry + HMAC token + bunle CDN URL build,
  and replaced the legacy signed-URL `/render` endpoint.
- Both included filesystem migrations for already-rendered chapters.
  See the commit messages for the SQL applied.
