# Beta Deploy — macOS + Cloudflare Tunnel + Pages

Single-host Mac deploy. FastAPI + workers run locally; Cloudflare
Tunnel exposes the API; Cloudflare Pages serves the SPA. Chapter
uploads go browser → R2 inbox direct, never through the home upstream
or the tunnel.

```
                    ┌─► CF Pages (mangalocal.com, static SPA)
browser ─HTTPS──► CF edge
                    └─► tunnel ─► localhost:8000  (FastAPI + workers)
        │
        └─PUT(zip) ──► R2 inbox  ◄─GET──  worker (after upload-finalize)
```

---

## Prerequisites

- `cloudflared` installed via Homebrew.
- `wrangler` installed via Homebrew / npm.
- Both logged in (see below).
- Postgres running locally.
- `.venv` with `typoon` installed.

---

## 1. Production `.env`

All secrets live in `.env` at the repo root. `python-dotenv` loads it
automatically when `WorkingDirectory` is set in the LaunchAgent.

```bash
# Generate a stable JWT secret (do once, never rotate without coordinated logout)
JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
```

Minimum production `.env`:

```ini
TYPOON_ENV=production          # fail-fast if JWT_SECRET missing
DATABASE_URL=postgresql://typoon:typoon@localhost:5432/typoon

DISCORD_CLIENT_ID=<snowflake>
DISCORD_CLIENT_SECRET=<secret>
DISCORD_GUILD_ID=<snowflake>
DISCORD_BOOTSTRAP_ID=<user-snowflake>   # first admin

PUBLIC_API_URL=https://api.mangalocal.com
PUBLIC_WEB_URL=https://mangalocal.com
TRUSTED_HOSTS=api.mangalocal.com,mangalocal.com,localhost,127.0.0.1

JWT_SECRET=<86-char urlsafe token>

# Public render storage — HuggingFace + bunle CDN
PUBLIC_STORE_TYPE=huggingface
HF_REPO=nghyane/mcz-cdn
HF_CDN_PREFIX=https://927251094806098001.discordsays.com/cdn/t
HF_TOKEN=<hf token from ~/.cache/huggingface/token>

# Browser-direct chapter uploads — bypasses the home upstream
# bandwidth by PUTting a multipart-zipped chapter straight to an
# S3-compatible inbox bucket. Any S3-compatible provider works: R2,
# AWS S3, Backblaze B2, MinIO, Wasabi.
#
# The server reads only ENV — no secret in the toml. Bucket/CORS/
# lifecycle wiring lives in section 5.5.
INBOX_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
INBOX_S3_BUCKET=mangalocal-uploads
INBOX_S3_REGION=auto
INBOX_S3_ACCESS_KEY_ID=<inbox access key>
INBOX_S3_SECRET_ACCESS_KEY=<inbox secret>
INBOX_S3_PREFIX=tmp/
```

`TYPOON_ENV=production` makes the engine raise at startup if `JWT_SECRET`
is empty, preventing silent session invalidation on restart.

---

## 2. Cloudflare Tunnel

### Login (once per machine)

```bash
cloudflared tunnel login
# Opens browser → select zone (mangalocal.com)
# Writes ~/.cloudflared/cert.pem
```

### Create tunnel + DNS

```bash
cloudflared tunnel create typoon-beta
# Writes ~/.cloudflared/<uuid>.json

cloudflared tunnel route dns typoon-beta api.mangalocal.com
# Creates CNAME api.mangalocal.com → <uuid>.cfargotunnel.com
```

### Config — `~/.cloudflared/config.yml`

```yaml
tunnel: 9b27654b-f114-4e9d-b3ac-127c94611dab
credentials-file: /Users/nghiahoang/.cloudflared/9b27654b-f114-4e9d-b3ac-127c94611dab.json

ingress:
  - hostname: api.mangalocal.com
    service: http://localhost:8000
    originRequest:
      connectTimeout: 30s
      tcpKeepAlive: 30s
      keepAliveTimeout: 90s   # > SSE heartbeat (15s) + buffer
      keepAliveConnections: 100
      noHappyEyeballs: true   # localhost resolves to one stack

  - service: http_status:404
```

### LaunchAgent — `~/Library/LaunchAgents/com.cloudflare.typoon-beta.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.cloudflare.typoon-beta</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/cloudflared</string>
    <string>--config</string><string>/Users/nghiahoang/.cloudflared/config.yml</string>
    <string>--no-autoupdate</string>
    <string>tunnel</string><string>run</string><string>typoon-beta</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key>
    <string>/Users/nghiahoang/Library/Logs/cloudflared-typoon.log</string>
  <key>StandardErrorPath</key>
    <string>/Users/nghiahoang/Library/Logs/cloudflared-typoon.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>/Users/nghiahoang</string>
  </dict>
</dict>
</plist>
```

---

## 3. Cloudflare Pages (web SPA)

### Login

```bash
wrangler login
```

### Build

```bash
cd web
# .env.production must exist:
echo "VITE_API_URL=https://api.mangalocal.com" > .env.production
bun install --frozen-lockfile
bun run build          # outputs web/dist/
```

### Create project + deploy

```bash
wrangler pages project create mangalocal-web --production-branch=main
wrangler pages deploy web/dist --project-name=mangalocal-web \
  --branch=main --commit-dirty=true
```

### Attach custom domains (via CF API — wrangler CLI lacks this subcommand)

```bash
ACCOUNT_ID="11a58966d1289d0387eebba638a19cc3"
TOKEN=$(grep '^oauth_token' ~/Library/Preferences/.wrangler/config/default.toml \
        | cut -d'"' -f2)

for domain in mangalocal.com www.mangalocal.com; do
  curl -sS -X POST \
    "https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/pages/projects/mangalocal-web/domains" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    --data "{\"name\":\"$domain\"}"
done
```

### DNS records for Pages (via tunnel API token which has DNS write)

```bash
ZONE_ID="e0f3e8ccb935ffffa988eee50b3208a0"
TUNNEL_TOKEN="cfut_..."   # from ~/.cloudflared/cert.pem JSON

for name in "@" "www"; do
  curl -sS -X POST \
    "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" \
    -H "Authorization: Bearer $TUNNEL_TOKEN" \
    -H "Content-Type: application/json" \
    --data "{\"type\":\"CNAME\",\"name\":\"$name\",\"content\":\"mangalocal-web.pages.dev\",\"ttl\":1,\"proxied\":true}"
done
```

### Redeploy after code changes

```bash
cd web && bun run build && cd ..
wrangler pages deploy web/dist --project-name=mangalocal-web \
  --branch=main --commit-dirty=true
```

---

## 4. FastAPI + Worker LaunchAgents

Both agents use `WorkingDirectory` pointing at the repo root so `.env`
is loaded automatically. `ThrottleInterval=30` prevents restart-loops
on bad config.

### `~/Library/LaunchAgents/com.typoon.api.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.typoon.api</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/nghiahoang/Dev/MANGA/ComicScan/v2/.venv/bin/typoon</string>
    <string>api</string>
  </array>
  <key>WorkingDirectory</key>
    <string>/Users/nghiahoang/Dev/MANGA/ComicScan/v2</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key><false/>
    <key>Crashed</key><true/>
  </dict>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key>
    <string>/Users/nghiahoang/Library/Logs/typoon-api.log</string>
  <key>StandardErrorPath</key>
    <string>/Users/nghiahoang/Library/Logs/typoon-api.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>/Users/nghiahoang</string>
    <key>PATH</key>
      <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
</dict>
</plist>
```

### `~/Library/LaunchAgents/com.typoon.worker.plist`

Same structure; replace `ProgramArguments` with:

```xml
  <key>ProgramArguments</key>
  <array>
    <string>/Users/nghiahoang/Dev/MANGA/ComicScan/v2/.venv/bin/typoon</string>
    <string>work</string>
    <string>--role</string><string>full</string>
    <string>--concurrency</string><string>3</string>
  </array>
```

And logs to `typoon-worker.log`.

### Load / manage

```bash
# Load all (first time or after editing plist)
for label in com.typoon.api com.typoon.worker com.cloudflare.typoon-beta; do
  launchctl load ~/Library/LaunchAgents/$label.plist
done

# Force kill + restart (launchd respawns after ThrottleInterval)
UID_=$(id -u)
launchctl kickstart -k gui/$UID_/com.typoon.api
launchctl kickstart -k gui/$UID_/com.typoon.worker
launchctl kickstart -k gui/$UID_/com.cloudflare.typoon-beta

# Status
launchctl list | grep -E "typoon|cloudflare"

# Disable permanently (won't start at login)
launchctl unload -w ~/Library/LaunchAgents/com.typoon.api.plist

# Logs
tail -f ~/Library/Logs/typoon-api.log
tail -f ~/Library/Logs/typoon-worker.log
tail -f ~/Library/Logs/cloudflared-typoon.log
```

---

## 5. Storage

Render archives live on HuggingFace (`HF_REPO`), served via the bunle
CDN through the Discord Activity proxy (`HF_CDN_PREFIX`). The SPA runs
cross-origin (CF Pages at `mangalocal.com` ≠ API at
`api.mangalocal.com`), so a public CDN is required.

Render URL pattern:

```
https://927251094806098001.discordsays.com/cdn/t/render/<key>.bnl?v=<ts>
```

---

## 5.5 Inbox bucket (browser-direct upload)

Recommended default: Cloudflare R2 (free egress to Cloudflare, free
10 GB storage). Any S3-compatible endpoint works the same way.

### Create bucket + API token (R2 dashboard)

1. **R2 → Create bucket** → name `mangalocal-uploads`. Region `auto`.
2. **R2 → API Tokens → Create** → permission `Object Read & Write`,
   scope to the bucket. Save the access key id + secret. Copy
   `Account ID` from R2 home → S3 endpoint is
   `https://<account>.r2.cloudflarestorage.com`.

### CORS — required

Browser PUT responses must surface `ETag` so the SDK can pass it back
to `upload-finalize`. Without `ExposeHeaders: ETag` the SDK fails with
`Missing ETag — kiểm tra CORS bucket`.

R2 bucket → Settings → CORS Policy:

```json
[
  {
    "AllowedOrigins": [
      "https://mangalocal.com",
      "https://927251094806098001.discordsays.com",
      "chrome-extension://<extension-id>"
    ],
    "AllowedMethods":  ["PUT"],
    "AllowedHeaders":  ["*"],
    "ExposeHeaders":   ["ETag"],
    "MaxAgeSeconds":   3600
  }
]
```

The DA origin (`*.discordsays.com`) is required because users running
the SPA inside the Discord Activity iframe PUT from that origin, not
from `mangalocal.com`.

### Lifecycle — sweep aborted uploads

R2 bucket → Settings → Lifecycle Rules → Add rule:

- **Prefix**: `tmp/` (matches `INBOX_S3_PREFIX`).
- **Action**: delete after `1 day`.
- **Multipart cleanup**: abort incomplete multipart after `1 day`.

The engine deletes successful uploads inline after ingest; the rule
catches anything that slipped through (browser closed mid-upload,
crash before `upload-abort`).

---

## 6. DNS records summary

| Record | Type | Target | Proxied |
|---|---|---|---|
| `api.mangalocal.com` | CNAME | `<tunnel-uuid>.cfargotunnel.com` | ✓ |
| `mangalocal.com` | CNAME | `mangalocal-web.pages.dev` | ✓ |
| `www.mangalocal.com` | CNAME | `mangalocal-web.pages.dev` | ✓ |

`api.mangalocal.com` is created by `cloudflared tunnel route dns` and
must not be edited manually.

---

## 7. Probe checklist

Run after any deploy or restart:

```bash
# API
curl -sS https://api.mangalocal.com/api/healthz          # 200 {"ok":true}
curl -sS https://api.mangalocal.com/api/projects          # 401 (auth gate)

# CORS — main API
curl -sS -X OPTIONS https://api.mangalocal.com/api/projects \
  -H "Origin: https://mangalocal.com" \
  -H "Access-Control-Request-Method: GET" \
  -o /dev/null -w "%{http_code}\n"                        # 200

# Upload init (auth-gated; expects 401 without token)
curl -sS -X POST https://api.mangalocal.com/api/projects/0/chapters/upload-init \
  -H "Content-Type: application/json" \
  --data '{"byte_size":1}' \
  -o /dev/null -w "%{http_code}\n"                        # 401

# Host header rejection
curl -sS https://api.mangalocal.com/api/healthz \
  -H "Host: evil.com" -o /dev/null -w "%{http_code}\n"   # 403

# Web SPA
curl -sS https://mangalocal.com/ -o /dev/null -w "%{http_code}\n"  # 200

# Render archive Range (bunle reader through HF CDN)
curl -sS -H "Range: bytes=0-15" \
  "https://927251094806098001.discordsays.com/cdn/t/render/<locator>.bnl" \
  -o /dev/null -w "%{http_code}\n"                        # 206

# Inbox CORS — must expose ETag on PUT preflight
curl -sS -X OPTIONS "$INBOX_S3_ENDPOINT/$INBOX_S3_BUCKET/probe" \
  -H "Origin: https://mangalocal.com" \
  -H "Access-Control-Request-Method: PUT" \
  -i 2>&1 | grep -i 'access-control-expose-headers'      # contains ETag
```

---

## 8. Middleware added for production

These were added during beta setup and are active in the codebase:

| Middleware | Purpose |
|---|---|
| `TrustedHostMiddleware` | Reject requests with unexpected `Host` header |
| `ProxyHeadersMiddleware` | Trust `X-Forwarded-For/Proto` from CF Tunnel |
| SSE `Cache-Control: no-cache, no-transform` | Prevent CF buffering event stream |
| `LocalArtifactStore(api_origin=...)` | Absolute URLs for cross-origin SPA |

`TRUSTED_HOSTS` env var overrides the auto-derived list from
`PUBLIC_API_URL` + `PUBLIC_WEB_URL`.

---

## 9. Discord Activity (DA) setup

App ID = `927251094806098001` (same as `DISCORD_CLIENT_ID`).

### Discord Developer Portal (one-time)

1. [discord.com/developers](https://discord.com/developers) → app `927251094806098001`
2. **Activities → Settings** — enable the activity, enable Web/iOS/Android as needed.
3. **URL Mappings** (under Activities → Settings):

   | Prefix  | Target                        |
   |---------|-------------------------------|
   | `/`     | `mangalocal.com`              |
   | `/api`  | `api.mangalocal.com/api`      |
   | `/files`| `api.mangalocal.com/files`    |

   > Target must include the path prefix so Discord does not strip it on forward.

4. **OAuth2 → Redirects** — add `https://127.0.0.1` (SDK placeholder URI).

### How it works

- SPA detects DA via `window.location.hostname.endsWith('.discordsays.com')`.
- In DA, `API_BASE = ''` — all fetch calls use relative paths (`/api/...`).
- Discord proxy forwards `/api/*` → `https://api.mangalocal.com/api/*` per URL Mapping.
- Login is automatic: SDK `authorize` command triggers Discord's native consent UI,
  code is exchanged via `/api/auth/discord/exchange` with `redirect_uri = https://127.0.0.1`.
- `TRUSTED_HOSTS` includes `*.discordsays.com` to allow Host header from Discord proxy.

### `.env` additions required

```ini
TRUSTED_HOSTS=api.mangalocal.com,mangalocal.com,localhost,127.0.0.1,*.discordsays.com
```

### `web/.env.production` additions required

```ini
VITE_DISCORD_CLIENT_ID=927251094806098001
```

---

## 10. Known issues / TODO

- **Postgres auto-start**: if Postgres is slow to start after reboot,
  the worker LaunchAgent exits and retries after 30s. Add
  `brew services start postgresql@17` to a login item or a separate
  LaunchAgent with `WaitForDependency` if this is a problem.
- **Log rotation**: `~/Library/Logs/typoon-*.log` grows unbounded.
  Add `newsyslog` config or a cron `logrotate` entry for production.
