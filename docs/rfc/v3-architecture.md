# Typoon v3 — Architecture & Migration Plan

Status: Design — pending implementation
Owner: nghiahoang
Date: 2026-05-20

---

## 1. Vision & Scope

**Typoon = manga translation tool, not a manga platform.**

Users bring content (URL or upload), get a translated archive back. The
server is a stateless compute service plus a thin sync layer for cross-device
personal data. The server never hosts public manga catalogs.

### What Typoon IS
- A drag-drop translation tool (zip / chapter URL → translated BNL archive)
- A personal reader with browse adapters (search public sources, read raw,
  trigger translation per chapter)
- A cross-device sync service for personal library / glossary

### What Typoon IS NOT
- A manga hosting site
- A community translation platform
- A discovery index

---

## 2. Why this change

Current state (v2) is over-engineered:

- ~30 D1 tables across 6 concerns (work/material/library/translation/glossary/memory)
- ~50 REST endpoints, many duplicated (per-work + per-material upload init)
- 4 community voting / linking tables that nobody uses
- Server stores reading history → DMCA + privacy liability
- Two storage paradigms competing: server-canonical catalog vs client manifest

Effect:
- Schema churn blocks features
- Endpoint mismatch (client sends X, server expects X+Y+Z) → constant 400s
- High legal risk: server is the system of record for "what users read"
- Hard to onboard new devs / agents

v3 reduces to:
- 7 D1 tables, 1 concern each
- ~13 endpoints
- Server = compute + auth + sync
- Client = source of truth for catalog/library/history
- Per-user quota model, ephemeral 7-day R2 lifecycle

---

## 3. Domain model

### 3.1 Entities (server-owned)

```
User                  — Discord-authenticated identity (+ tier from Discord role)
ApiToken              — programmatic access tokens (Supporter+ only)
Job                   — one translation run (input zip + context → output archive)
Consume               — billing ledger row per job (immutable)
UserDataSnapshot      — cloud-backup blob (1 row/user, JSON snapshot)
Report                — abuse report (admin moderation queue)
```

### 3.1.1 Stateless context-aware translation

Each Job is one chapter, but translation quality requires consistency
across chapters of the same manga — character names, terminology,
address pairs, narrative voice.

The server stays stateless: it doesn't track which chapters belong
together. **Context is owned by the client**, carried into each job,
updated by the pipeline, returned with the output.

```
Client (IndexedDB):
  Work {
    id, title, source_lang, target_lang,
    context: WorkContext { characters, glossary, address, style }
  }

POST /jobs {
  byte_size, source_lang, target_lang,
  context: WorkContext (gzip+base64 inline, optional)
}
  → server writes ctx/{job_id}/input.json to R2 (7d lifecycle)
  → returns presigned upload URLs

[upload + start]

Pipeline brief stage:
  - read R2 ctx/{job_id}/input.json (if exists)
  - process chapter → extract new context
  - merge: full context = input + new findings
  - write ctx/{job_id}/output.json

GET /jobs/:id (state='done') →
  archive_url + context_out_url

Client downloads context_out, replaces work.context
Client syncs work via /sync (cloud backup, cross-device)
```

**Properties:**
- Server has no `works` table; jobs are independent rows
- Two users translating the same manga → two independent Works
  → different contexts (personal naming, glossary tweaks)
- Lost device + no sync → lost context (mitigated by /sync + export)
- First chapter of a new Work → empty context, brief builds from scratch
- Subsequent chapters → carry context forward, brief only adds new entries

**Context format on the wire:**
- JSON.stringify(WorkContext) → gzip → base64
- Inline in POST /jobs body (typical 5-30KB)
- Size cap: 256KB compressed (rejected if larger; means Work has grown
  beyond reasonable context — client should prune old/unused entries)

### 3.2 Entities (client-owned, IndexedDB)

```
Work          — { id, title, source_lang, target_lang, context: WorkContext,
                  cover_url?, source?, upstream_ref?, status, created_at,
                  updated_at, deleted? }
                Persistent grouping. Stable id (client-generated nanoid).
                Context survives across jobs in the same work.

WorkContext   — { characters: Character[], glossary: GlossaryEntry[],
                  address: AddressPair[], style_notes: string,
                  version: number }
                Authoritative copy lives in IndexedDB. Server only ever
                sees ephemeral snapshots in R2.

LibraryItem   — { work_id, status, added_at, updated_at }
                Read-tracking + bookmark layer; references Work by id.

HistoryItem   — { work_id, chapter_ref, last_read_at, page }

JobRef        — { job_id, work_id, chapter_ref, archive_url, state,
                  expires_at }
                Mirror of server jobs, persists past R2 7d expiry so the
                reader can prompt "Tải về trước khi hết hạn".

Settings      — { theme, reader_mode, ... }
```

Server never sees these. Sync is opaque blob (JSON.stringify(everything)).

### 3.3 Source adapters (client-only)

```
SourceAdapter interface:
  id, name, languages, nsfw
  search(query) → Hit[]
  getDetail(ref) → Detail
  getChapters(ref) → Chapter[]
  getPages(chapterRef) → Page[]
  resolvePageUrl(page) → string (proxied via /cdn/c/)
```

Adapters: mangadex, otruyen, hitomi, ehentai, hentaifox, happymh (existing).
New adapters = client-only PR, no server change.

### 3.4 WorkContext — transport DTO

The shape exchanged between client and server when running a job in
the context of an existing Work. Authoritative copy lives in IndexedDB.

```ts
interface WorkContext {
  version:     number;         // monotonic counter (client-bumped on merge)
  source_lang: string;
  target_lang: string;

  characters:  {
    name:         string;       // canonical source-lang name
    target_name:  string;       // localized name
    aliases?:     string[];     // other source-lang spellings seen
    gender?:      "male" | "female" | "unknown";
    role?:        string;       // e.g. "protagonist", "antagonist"
    voice?:       string;       // 1-line style hint for translator
  }[];

  glossary:    {
    source_term: string;
    target_term: string;
    notes?:      string;
  }[];

  address:     {
    speaker:  string;            // character name
    listener: string;            // character name
    pair:     string;            // e.g. "anh/em", "tớ/cậu"
  }[];

  style_notes: string;           // free-form, ≤ 500 chars
}
```

Wire format: `JSON.stringify(WorkContext)` → gzip → base64
- Inline in `POST /jobs` body as `context_gz_b64`
- 256KB hard cap (compressed); over → 413 Payload Too Large

Client is the source of truth. Server only sees ephemeral R2 snapshots
(`ctx/{job_id}/input.json.gz` and `ctx/{job_id}/output.json.gz`).

---

## 4. Server architecture

### 4.1 Tier system (Discord-role-driven)

User tier is derived from Discord roles in the Typoon community guild on
every `/auth/discord/exchange`. No manual upgrade flow, no payment
integration in v3 — admins assign roles via Discord UI after
ko-fi / Patreon confirmation.

**Tiers** (`workers/api/src/lib/tiers.ts`):

| Tier | Monthly chapters | Max pages/ch | Concurrent | Sync quota | API tokens | Priority |
|------|-----------------:|-------------:|-----------:|-----------:|:----------:|---------:|
| Free | 20 | 200 | 2 | 1 MB | ✗ | 0 |
| Supporter | 100 | 200 | 3 | 5 MB | ✓ | 1 |
| Pro | 500 | 200 | 5 | 20 MB | ✓ | 2 |
| Unlimited | 99999 | 500 | 10 | 100 MB | ✓ | 3 |

**Tier resolution**:
- Map Discord role IDs → tier IDs via `DISCORD_ROLE_TIER_MAP` secret (JSON)
- Highest priority wins when multiple tier roles present
- No tier role → `free`
- Tier cached in `users.tier_id`, also embedded in JWT claims (`tier_id`)
- Re-sync on every login (JWT TTL 7d → max delay 7 days between role change
  and reflection)

**Admin** is a separate flag (`is_admin`), driven by `ADMIN_ROLE_ID`
membership, orthogonal to tier ladder.

**Quota check** combines:
- `COUNT(consumes WHERE counted=1 AND month=current)` vs `tier.monthly_chapters`
- `COUNT(jobs WHERE state IN active)` vs `tier.concurrent_jobs`
- `byte_size` estimate / 1MB vs `tier.max_pages_per_chapter` (early reject)

**Failed-job billing**:
- Reject at `prepare` (>max_pages, decode error) → not counted
- Fail after `prepare` (LLM error, typeset crash) → counted (resource spent)
- User `DELETE /jobs/:id` before `start` → not counted
- User `DELETE /jobs/:id` after `start` → counted

### 4.2 Workers

```
typoon-api          Hono app — auth, jobs, sync, glossary, cdn proxy
typoon-pipeline     Workflows — orchestrates stages (existing, rename only)
typoon-media        Container — prepare/typeset-pack (existing)
typoon-scan         Container — bubble detect + OCR (existing)
typoon-brief        LLM — context generation (existing)
typoon-translate    LLM — translation (existing)
typoon-inpaint      Container — Lama inpaint (existing)
typoon-typeset-pack Container — render + BNL pack (existing)
```

**8 workers** — keep as-is. They already work and are independently
deployable. No consolidation in v3.

### 4.3 D1 schema (clean slate)

```sql
-- Identity
users (
  id, discord_id, display_name, avatar_url, email,
  preferred_target_lang TEXT,
  tier_id             TEXT NOT NULL DEFAULT 'free',
  tier_synced_at      TEXT,
  discord_roles       TEXT,                -- JSON snapshot for debug
  created_at, last_login_at
)

identities (
  id, user_id FK, provider, external_id, metadata,
  UNIQUE (provider, external_id)
)

api_tokens (
  id, user_id FK, name, token_hash, prefix, scopes JSON,
  last_used, created_at, revoked_at
)

-- Translation
jobs (
  id, user_id FK,
  source_lang, target_lang,
  state TEXT CHECK IN ('init','uploading','pending','running','done','error','expired'),
  progress_stage, progress_index, progress_total,
  zip_key, archive_key, page_count,
  estimated_pages INTEGER,            -- quota reserve at create time
  error_message,
  created_at, started_at, finished_at,
  expires_at TEXT NOT NULL            -- created_at + 7d
)
INDEX (user_id, created_at DESC)
INDEX (expires_at) WHERE state != 'expired'

chapter_consumes (
  id, user_id FK, job_id FK,
  page_count INTEGER NOT NULL,             -- actual pages, analytics only
  counted    INTEGER NOT NULL DEFAULT 1,   -- 1=count toward quota; 0=waived
  created_at
)
INDEX (user_id, created_at DESC)

-- Sync
user_data (
  user_id PK FK,
  payload TEXT NOT NULL,              -- JSON blob
  version INTEGER NOT NULL,           -- monotonic, for optimistic concurrency
  byte_size INTEGER NOT NULL,
  device_id TEXT,                     -- last writer hint
  updated_at
)

-- Moderation
reports (
  id, reporter_id FK, job_id FK,
  reason, status, created_at, resolved_at
)
```

**7 tables total.**

### 4.4 R2 layout

```
raw/{job_id}/source.zip              ← 7-day lifecycle
prepared/{job_id}/{page}.webp        ← 7-day lifecycle
scan/{job_id}/{page}.msgpack         ← 7-day lifecycle
mask/{job_id}/{page}.png             ← 7-day lifecycle
inpaint/{job_id}/{page}.webp         ← 7-day lifecycle
ctx/{job_id}/input.json.gz           ← 7-day lifecycle (client-supplied context)
ctx/{job_id}/output.json.gz          ← 7-day lifecycle (brief-updated context)
archive/{job_id}/output.bnl          ← 7-day lifecycle
```

One bucket (`typoon-work`), one lifecycle rule covers all prefixes.

### 4.5 HTTP surface

```
# Auth (4)
POST  /auth/discord/exchange         body: { code, redirect_uri }
POST  /auth/refresh                  re-sync tier from Discord role + reissue JWT
GET   /auth/me                       → SessionUser (incl. tier)
PATCH /auth/me/preferences           body: { preferred_target_lang? }

# Jobs (8)
POST   /jobs                         body: { byte_size, source_lang, target_lang?,
                                              context_gz_b64? }
                                     → { job_id, parts:[{number,url}], part_size,
                                          context_uploaded: boolean, expires_in }
                                     (quota reserve, R2 multipart init, presigned URLs,
                                      optional context written to ctx/{job_id}/input.json)
POST   /jobs/:id/start               body: { parts:[{number,etag}] }
                                     → { state }
                                     (complete R2 multipart, trigger pipeline)
GET    /jobs/:id                     → JobDetail { ..., archive_url?, context_out_url? }
GET    /jobs/:id/ws                  → WebSocket progress stream
GET    /jobs/:id/download            → 302 redirect to presigned R2 GET
DELETE /jobs/:id                     → cleanup R2 + DB, release quota if pending
GET    /me/jobs                      → paginated job list (last 7d)
GET    /me/quota                     → { tier, used_chapters, active_jobs, reset_at }

# Sync (3)
GET    /sync                         → { payload, version, updated_at }
POST   /sync                         body: { payload, base_version }
                                     → 200 { version } | 409 { version: server_version }
DELETE /sync                         → wipe blob (logout-all-devices)

# Moderation (1)
POST   /reports                      body: { job_id?, reason }

# Proxy (1)
ANY    /cdn/c/<host><path>           → CORS proxy for source adapters
```

**~17 endpoints total** (incl. CORS + reports + auth/refresh).

### 4.6 Pipeline rename + context plumbing

```diff
- interface PipelineParams { chapter_id, draft_id, source_lang, target_lang, zip_key }
+ interface PipelineParams {
+   job_id, source_lang, target_lang, zip_key,
+   context_in_key?: string,   // R2 ctx/{job_id}/input.json.gz (if client supplied)
+ }

- K.prepared(chapter_id, i)  → "prepared/{chapter_id}/{i}.webp"
+ K.prepared(job_id, i)      → "prepared/{job_id}/{i}.webp"

- BriefService.briefChapter({ chapter_id, scan_keys, storyboard_keys, ... })
+ BriefService.briefJob({
+   job_id, scan_keys, storyboard_keys,
+   context_in_key?: string,   // pass-through from PipelineParams
+ }) → { index_key, context_out_key, ... }

- API.finalize({ chapter_id, draft_id, archive_key, page_count, scan_keys, mask_keys })
+ API.finalize({ job_id, archive_key, page_count, context_out_key })

- API.notifyProgress({ draft_id, stage, index, total })
+ API.notifyProgress({ job_id,  stage, index, total })

- API.notifyError({ draft_id, stage, message })
+ API.notifyError({ job_id,  stage, message })
```

Brief stage behaviour:

- If `context_in_key` present: gunzip, parse `WorkContext` JSON, use as
  seed for Phase 1 (skip vision calls for characters / glossary / address
  if seed is rich enough; otherwise augment).
- Internal artifacts (brief.json, brief-prose.txt, brief-chars.json,
  brief-glossary.json, brief-address.json, brief-notes.json,
  brief-noise.json) keep their current per-job R2 paths; downstream
  consumers (translate, typeset) are unaffected.
- After Phase 1+2 complete, serialize the merged `WorkContext` (chars,
  glossary, address, style_notes) into `ctx/{job_id}/output.json.gz` —
  one file the client downloads to refresh its persistent context.
- Return `context_out_key` to the workflow → callback writes it to job row.

`WorkContext` is a **transport DTO** for client ↔ server. Internal brief
artifacts stay separate so we don't refactor translate/typeset.

Client downloads `context_out_url` from job response, replaces
`work.context` in IndexedDB, syncs via `/sync`.

Pipeline internal logic unchanged otherwise. Only param names + R2
prefixes change.

---

## 5. Client architecture

### 5.1 Storage tiers

```
Tier 0: Memory (React state, TanStack Query)
Tier 1: IndexedDB (Dexie) — primary store
        ├ library
        ├ history
        ├ glossary
        ├ job_refs (translated chapter mapping)
        └ settings
Tier 2: Server sync (POST /sync, debounced) — cloud backup
Tier 3: Manual export (Settings → "Tải về backup .json")
```

### 5.2 Module layout

```
web/src/
  features/
    auth/               session, discord OAuth
    sync/               SyncManager, LWW merge
    library/            useLibrary, LibraryCard (IndexedDB-backed)
    explore/            search across adapters (existing, no server)
    work/               work hub (renders adapter detail)
    reader/             pager/strip view (existing, mostly unchanged)
    browse/             source adapters (existing, no change)
    jobs/               useJobs, useJob, JobCard, ProgressIndicator
    translate/          "Dịch chương này" button + flow
    glossary/           CRUD UI for glossary
    settings/           preferences + quota + backup
  shared/
    api/                fetch wrappers, types
    db/                 Dexie schema + queries
    discord/            DA SDK
    ui/                 design system
  routes/               TanStack Router file routes
```

### 5.3 IndexedDB schema (Dexie v1)

```ts
db.version(1).stores({
  library:  '&id, source, status, updated_at',
  history:  '[source+upstream_ref+chapter_ref], last_read_at',
  glossary: '++id, source_lang, target_lang, source_term',
  job_refs: '++job_id, [source+upstream_ref+chapter_ref], state, expires_at',
  settings: '&key',
})
```

`id` for library = `${source}:${upstream_ref}` (composite key).

### 5.4 Sync mechanics

```
On boot:
  1. Hydrate session (existing)
  2. If authenticated: SyncManager.pullOnStartup()
     - GET /sync
     - if remote.version > local.version: merge LWW, write to Dexie
     - cache server version

On any IndexedDB mutation:
  - SyncManager.schedulePush() (2s debounce)

On push:
  - Collect snapshot from Dexie
  - POST /sync { payload, base_version }
  - 200 → update local server_version
  - 409 → pull + merge + retry once
  - Other error → log, retry on next mutation

On logout:
  - sendBeacon /sync (best-effort final push)
  - Clear Dexie (option: "Xóa data local")

On window blur:
  - Flush pending push (sendBeacon)
```

LWW merge per item by `updated_at`. Tombstones (`deleted: true`) for deletes.

### 5.5 Translation flow

```
User reads chapter (Reader, raw mode):
  → fetch pages via SourceAdapter.getPages() + CORS proxy
  → render pages

User clicks "Dịch chương này":
  1. Reader collects page image URLs
  2. Client builds upload zip in-browser (fflate)
     OR: future endpoint POST /jobs/from-urls { urls[] } (server-side fetch)
  3. POST /jobs { byte_size, source_lang, target_lang? }
     → presigned upload URLs
  4. Client PUT parts to R2
  5. POST /jobs/:id/start
  6. WebSocket /jobs/:id/ws → live progress
  7. Job done → archive_url in response
  8. Store in IndexedDB job_refs:
     { job_id, source, upstream_ref, chapter_ref, archive_url, expires_at }
  9. Reader detects translated archive available → switches to BNL viewer
  10. SyncManager debounce-pushes job_refs to /sync

Re-open chapter later:
  → IndexedDB lookup job_refs[source, upstream_ref, chapter_ref]
  → if found + not expired: load BNL archive directly
  → if expired: show "Dịch lại" button
```

### 5.6 Quota enforcement

```
Before POST /jobs:
  - Client check IndexedDB rough page count
  - GET /me/quota → { used, total, reset_at }
  - If reserve would exceed: show error before upload

Server hard check on POST /jobs:
  - SUM(consumes) for current month
  - + active jobs (state in init/uploading/pending/running).estimated_pages
  - vs user.monthly_quota_pages
  - Return 429 with details if over
```

---

## 6. Migration plan

### Phase 0: Plan freeze + cleanup branch
- Create branch `feat/v3-job-architecture`
- Commit this design doc
- **DROP production D1 + R2 contents** (no real users yet — confirmed)

### Phase 1: Schema reset
**Deliverables**:
- `workers/api/schema.sql` rewritten (7 tables only)
- `wrangler d1 execute typoon-db --remote --file schema.sql`
- R2 lifecycle rule applied via `wrangler r2 bucket lifecycle add typoon-work --prefix "" --expire-days 7`
- Old D1 dropped

**Files touched**: 1 schema + commands
**Verification**: `wrangler d1 execute typoon-db --remote --command "SELECT name FROM sqlite_master WHERE type='table'"` shows 7 tables

### Phase 2: Server rewrite
**Deliverables**:
- `workers/api/src/routes/jobs.ts` (new, ~250 LOC)
- `workers/api/src/routes/sync.ts` (new, ~80 LOC)
- `workers/api/src/routes/auth.ts` (trim, keep)
- `workers/api/src/routes/me.ts` (trim → only `/me/jobs`, `/me/quota`, `/me/preferences`)
- `workers/api/src/routes/glossary.ts` (delete — moved into sync payload)
- `workers/api/src/store/` reset:
  - keep: `users.ts`, `db.ts` (errors)
  - delete: `works.ts`, `materials.ts`, `chapters.ts`, `community.ts`,
            `library.ts`, `memory.ts`, `glossary.ts`, `translations.ts`,
            `drafts.ts`, `quota.ts` (rewrite slim), `admin.ts`
  - new: `jobs.ts`, `consumes.ts`, `sync.ts`
- `workers/api/src/index.ts` mount only new routes
- `workers/api/src/types.ts` strip ApiWork, ApiMaterial, etc; add ApiJob
- `workers/api/src/rpc/pipeline-callback.ts` accept `job_id`
- Delete `workers/api/src/do/translation-status.ts` → replace with new
  `workers/api/src/do/job-status.ts` (same DO concept, different schema)

**Files touched**: ~25 in `workers/api/`
**Verification**:
- `tsc --noEmit` clean
- `wrangler deploy` succeeds
- `curl /api/auth/config`, `curl /api/me/quota` with token return 200

### Phase 3: Pipeline rename
**Deliverables**:
- `workers/pipeline/src/pipeline.ts` rename `chapter_id`/`draft_id` → `job_id`
- `workers/pipeline/src/services.ts` callback signature updates
- `workers/pipeline/src/http.ts` accept `job_id` instead
- `workers/shared/src/keys.ts` rename `K.prepared(chapter_id, ...)` → `K.prepared(job_id, ...)`
- Downstream workers (`media`, `scan`, etc) use updated K helpers — likely just
  parameter rename, no logic change

**Files touched**: ~10 across workers
**Verification**:
- `tsc --noEmit` clean in each worker
- Manual job submission end-to-end (POST /jobs → done state)

### Phase 4: Client data layer
**Deliverables**:
- `web/src/shared/db/schema.ts` (Dexie)
- `web/src/shared/db/queries.ts` (typed CRUD)
- `web/src/features/sync/SyncManager.ts`
- `web/src/features/sync/lww.ts` (merge logic + tests)
- `web/src/features/sync/hooks.ts` (`useSyncStatus`)
- `web/src/main.tsx` wire `SyncManager.pullOnStartup()` after auth

**Files touched**: ~6 new files in `web/src/`
**Verification**:
- Unit tests for LWW merge
- E2E: login → IndexedDB populated from /sync → mutate → debounce → server reflects

### Phase 5: Client feature rewrite (data hooks)
**Deliverables**:
- `web/src/features/library/` rewrite hooks → Dexie queries
- `web/src/features/work/useWorkData.ts` rewrite → composite from adapter (no /api/work)
- `web/src/features/library/addManga/useImportToLibrary.ts` rewrite → Dexie put
- `web/src/features/jobs/` new module: `useJobs`, `useJob`, mutations
- `web/src/features/translate/useTranslateChapter.ts` new
- `web/src/features/reader/useReader.ts` patch — resolve translated archive
  from Dexie job_refs instead of /api/translations

**Files touched**: ~15 in `web/src/features/`
**Verification**:
- Library page renders from IndexedDB
- Work hub loads from source adapter (live fetch)
- Reader shows raw pages from adapter; clicking translate creates a job

### Phase 6: Strip legacy
**Deliverables**:
- Delete `web/src/features/{material,link}/` (work voting/linking)
- Delete `web/src/features/library/addManga/BlankCreateRow.tsx`
- Delete `web/src/features/work/{LinkSuggestionPanel,LinkSearchModal,...}.tsx`
- Delete `web/src/routes/admin.ops.tsx` (no admin UI in v3)
- Cleanup `web/src/shared/api/api.ts` (drop ~70% of endpoints)

**Files touched**: ~20 files (mostly deletes)
**Verification**:
- `npm run build` clean
- No broken imports

### Phase 7: UI redesign (LAST)
**Deliverables**:
- Redesign `/` (home) → drag-drop hero
- Redesign `/jobs` → activity feed
- Redesign `/settings` → preferences + quota chart + backup export
- Refresh component tokens / spacing
- Mobile pass

**Out of scope for THIS doc** — separate design doc.

---

## 7. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Discord OAuth credentials leak in dev | Low | High | `.dev.vars` in `.gitignore`, rotate periodically |
| User loses library on device wipe | Medium | Low | Sync to server + export backup file in Settings |
| Sync conflict (LWW loses data) | Low | Medium | Tombstones, version guard, manual conflict UI later |
| 5MB sync blob too small for power users | Low | Medium | Future: chunked sync (split library/history/glossary) |
| Per-user quota gaming (many small jobs) | Low | Low | Quota is per-page not per-job |
| LLM cost > revenue | Medium | High | Free tier 200pg/mo, paid tier later |
| Browse adapter breakage (source HTML change) | High | Low | Adapters are client-only, push fix as web deploy |
| R2 lifecycle deletes archive before user downloads | Low | Medium | 7d is generous; UI shows expires_at; "extend" feature later |

---

## 8. Non-goals (v3)

- Public manga catalog / discovery
- Community translations (shared drafts across users)
- Community glossary voting
- Material linking / splitting / voting
- Admin moderation UI (use D1 console for now)
- Mobile native app
- Browser extension (separate workstream, uses same /jobs API)

---

## 9. Success criteria

End of Phase 6:
- D1 has 7 tables, no `works`/`materials`/`library_*`/`community_*`
- API exposes ~16 endpoints
- Library + work hub + reader load without `/api/work` or `/api/library` calls
- Translation job E2E works (raw upload → archive download)
- Cross-device sync works (login on device B shows device A library)
- `npm run build` clean both `workers/api` and `web`

End of Phase 7:
- New homepage shipped
- Quota + backup UI in Settings
- Mobile responsive

---

## 10. Open questions

1. **Quota refill timing**: 1st of month UTC (chosen for simplicity)
2. **Tier role IDs**: need to create roles in Discord guild and populate
   `DISCORD_ROLE_TIER_MAP` secret. Discord guild = `1501760080134148197`.
3. **Tier downgrade behavior**: when user loses paid role mid-month, do we
   immediately revoke quota access or honor until reset? → propose honor
   until next 1st (no mid-month claw-back).
4. **Pro/Unlimited features beyond quota**: priority queue in pipeline,
   higher max pages/chapter (200 → 500), API tokens. Confirm priority impl
   actually wires into Workflow queue or defer to v3.1.
5. **Failed-job billing rule**: see section 4.1; charge only after `prepare`
   succeeds. Confirm acceptable.
6. **Sync encryption at rest**: defer to v3.1.
7. **Extension reuses /jobs API verbatim**? Yes, no separate endpoints.
