# Phase 1 — handoff state

This file is a snapshot of where Typoon stands. RFC-005 (Postgres-only)
is **complete** — SQLite, the in-process event bus, and the dual-mode
branching are all gone. Read this before opening the codebase.

## Where we are

- Postgres 17 (asyncpg) is the only database. DSN comes from
  `DATABASE_URL`; the engine refuses to start without it.
- Engine API: `typoon api` on `:8000` (configurable via `[server]`).
- Workers: `typoon work --role full` against the same Postgres. Roles
  `vision` / `llm` / `api` work too — the SQLite role guard is gone.
- Web: Vite dev on `:5173`, proxies `/api` and `/files` to `:8000`.
- Auth: Discord OAuth, JWT bearer, `users` + `identities` tables. SPA
  owns the OAuth redirect. Backend is a pure JSON exchanger.
- Artifacts: `~/.typoon/artifacts/` (LocalArtifactStore). Object
  storage (R2) is a future RFC; in Phase 1 the worker and API share
  the same disk.

## What works (verified)

- Project CRUD: create empty project + cover upload + delete.
- Chapter ingestion: PDF, CBZ/ZIP, multi-image upload via
  `POST /api/projects/{id}/chapters/upload`.
- Pipeline: prepare → scan → translate → render runs end-to-end.
- Bubble OCR list, manual translation edit (PATCH).
- Glossary CRUD with Postgres `tsvector` search.
- Project settings (title, description, target_lang).
- Chapter redo + delete.
- SSE event stream with disconnect cleanup, replay via `Last-Event-ID`,
  Postgres LISTEN/NOTIFY for live wakeup.
- Workers dashboard (queue depth per stage, derived via `FOR UPDATE
  SKIP LOCKED` claim).
- FTS search (LLM agent's `search_knowledge` tool — important for
  translation context). `glossary_search`, `search_briefs`, and
  `search_context` use `websearch_to_tsquery` so the agent passes
  natural strings.
- `/api/healthz` returns 200 + DB ping.
- Discord OAuth login (SPA-driven flow):
  - `GET /api/auth/config` — public client_id + guild branding (name, icon, invite). 503 if Discord Server Widget not enabled and gating is on.
  - `POST /api/auth/discord/exchange { code, redirect_uri }` — used by both web standalone and Discord Activity.
  - `GET /api/auth/me` — current user + guild branding.
  - `POST /api/auth/logout` — no-op stub for stateful logout when added.
  - SPA owns redirect_uri (`window.origin + /auth/callback`) and CSRF state (sessionStorage). No cookies, no engine HTML.
- Guild branding: name from Discord widget endpoint, icon hash cached from `/users/@me/guilds` on first member login. Engine fails loud (503) instead of silently substituting a placeholder name.
- Bootstrap admin via `DISCORD_BOOTSTRAP_ID` env.
- Guild gating via `DISCORD_GUILD_ID` env.

## What does not work yet

- Multi-host: code is ready (Postgres + DSN), but no VPS deployment
  has been set up. Tailscale + nginx + production Dockerfile come
  next.
- Object storage: artifacts only on the same disk as the API. Phase 2
  (R2 / HTTP relay) when the worker and API are on different hosts.

## Code map

```
typoon/
  adapters/          external systems
    artifact_store.py    LocalArtifactStore
    chapter_archive.py   prepared_key/render_key/masks_key + pack_and_upload
    event_bus.py         EventBus (Postgres LISTEN/NOTIFY) + EventHook
    projects.py          Projects (CRUD + ingest_chapter)
    ctx.py, loader.py
  api/
    app.py             FastAPI + middleware + static /files mount
    auth.py            JWT issue/verify, Discord token+user+guild fetchers
    deps.py            get_store, get_bus, get_paths, get_artifact_store,
                       require_user, require_admin
    models.py          Pydantic response shapes
    routes/
      _shared.py       require_project, require_chapter, chapter_out
      auth.py          /api/auth/{config,discord/exchange,me,logout}
      bubbles.py       /api/projects/{id}/chapters/{cid}/bubbles[/...]
      glossary.py      /api/projects/{id}/glossary[/...]
      pages.py         /api/projects/{id}/chapters/{cid}/pages/{idx}
      projects.py      project + chapter CRUD + cover upload + settings + brief
      search.py        /api/search?q&project_id&scope
      sse.py           /api/events?token=
      upload.py        /api/projects/{id}/chapters/upload
      workers.py       /api/workers
  cli/
    commands.py        typoon add/redo/work/api/export/status
  domain/              data types only (prepared, scan, translate, render)
  llm/                 IR + tool registry
  paths.py             ~/.typoon/* layout
  runs/events.py       Hook + LoggingHook + CompositeHook + Event types
  sources/
    constants.py       IMAGE_EXTS
    local.py           LocalSource
    upload.py          unpack_zip / unpack_pdf / write_image_files
  stages/              prepare, scan, translate, render
  storage/
    postgres.py        PostgresStore (asyncpg pool, single backend)
    schema.sql         full DDL applied at PostgresStore.open()
    store.py           Store protocol (the contract)
    __init__.py        re-exports
  vision/, workers/    pipeline workers + GPU runtime
  config.py            Config + AuthConfig (env-overridable)

web/src/
  components/
    AppLayout.tsx      auth gate + sidebar/header/main shell
    Cover.tsx          image with letter fallback + cache-bust on update
    CreateProjectDialog.tsx
    UploadChapterDialog.tsx
    GlossaryPanel.tsx
    SettingsPanel.tsx
    Header.tsx         user menu, search shortcut, workers indicator
    Modal.tsx, Toaster.tsx, Sidebar.tsx, ui.tsx, WorkersIndicator.tsx
  lib/
    api.ts             fetch wrapper, types, all REST methods
    auth.ts            token storage, useCurrentUser, OAuth URL builder,
                       state CSRF, exchangeCode
    chapter.ts         STATE table, stageLabel, chapterStats
    cn.ts, events.ts, time.ts
  routes/
    __root.tsx         skip AppLayout for /login + /auth/*
    auth.callback.tsx  Discord redirect handler
    login.tsx          Discord login button
    projects.tsx       layout for /projects/*
    projects.index.tsx project list
    projects.$projectId.tsx  detail + tabs
    settings.tsx       per-app settings stub
    index.tsx          redirect to /projects
```

## API surface

```
Auth:
  GET    /api/auth/config                       public client_id + guild branding (or 503)
  POST   /api/auth/discord/exchange             { code, redirect_uri } → { token }
  GET    /api/auth/me                           current user + guild branding
  POST   /api/auth/logout                       no-op stub

Projects:
  GET    /api/projects                          list
  POST   /api/projects                          create empty (form fields)
  GET    /api/projects/{id}
  DELETE /api/projects/{id}
  POST   /api/projects/{id}/cover               multipart image
  GET    /api/projects/{id}/settings
  PATCH  /api/projects/{id}/settings

Chapters:
  GET    /api/projects/{id}/chapters
  GET    /api/projects/{id}/chapters/{cid}
  DELETE /api/projects/{id}/chapters/{cid}
  POST   /api/projects/{id}/chapters/{cid}/redo
  GET    /api/projects/{id}/chapters/{cid}/brief
  POST   /api/projects/{id}/chapters/upload     multipart PDF/CBZ/ZIP/images

Bubbles:
  GET    /api/projects/{id}/chapters/{cid}/bubbles
  PATCH  /api/projects/{id}/chapters/{cid}/bubbles/{page}/{bubble}

Glossary:
  GET    /api/projects/{id}/glossary
  POST   /api/projects/{id}/glossary
  PATCH  /api/projects/{id}/glossary/{tid}
  DELETE /api/projects/{id}/glossary/{tid}

Misc:
  GET    /api/workers                           queue stats
  GET    /api/search?q=&project_id=&scope=     FTS over translations/briefs/glossary
  GET    /api/projects/{id}/chapters/{cid}/pages/{idx}   render asset (PNG/WEBP)
  GET    /api/events?token=                     SSE
  GET    /api/healthz                           liveness + DB ping
```

All `/api/*` except `/api/auth/{config,discord/exchange}` and the SSE
stream require `Authorization: Bearer <jwt>`. SSE accepts the token via
`?token=` because `EventSource` cannot set headers.

## Local setup

```bash
# 1. Postgres 17 — bring your own (host install, brew, managed). Once:
createuser -s typoon
createdb -O typoon typoon

# 2. Engine config — copy .env.example to .env and fill in:
#    DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, DISCORD_GUILD_ID,
#    DISCORD_BOOTSTRAP_ID
#    DATABASE_URL=postgresql://typoon:typoon@localhost:5432/typoon
cp .env.example .env

# 3. Engine
typoon api --port 8000      # one terminal
typoon work --role full     # another terminal

# 4. Web
cd web && npm install && npm run dev
```

Schema changes during dev (no Alembic in Phase 1):

```bash
dropdb typoon && createdb -O typoon typoon
```

Discord application setup (one-time per deployment):

1. https://discord.com/developers/applications → New Application
2. OAuth2 → Redirects → add `http://localhost:5173/auth/callback`
3. Reset Secret → paste into `DISCORD_CLIENT_SECRET` env or `~/.typoon/config.toml [auth]`
4. Copy Application ID → `DISCORD_CLIENT_ID`
5. Server Settings → Widget → **Enable Server Widget** (engine 503s
   without this when gating is on; the widget supplies guild name +
   icon + invite for the SPA branding)
6. Server Settings → Widget → **Invite Channel** → pick a public channel
   so widget exposes an instant invite (gating-failure UX)

## Open issues for the next thread

RFC-005 is done. Backlog (loosely ordered):

1. Discord Activity SDK integration (Phase 2).
2. Discord bot service (separate repo, consumes API).
3. Browser extension for quick-import (separate repo).
4. Object storage (R2) — only when API and worker are on different
   physical hosts.
5. Production Dockerfile for the API container.
6. nginx reverse proxy config.
7. Tailscale setup guide for worker ↔ API ↔ DB.
8. Bubble editor UI.
9. Chapter viewer (full-size manga page reader).
10. Search palette (⌘K) wired to `/api/search`.
11. Bot-issued per-user API tokens (when bot lands).
12. Quota enforcement (when payment lands).

## Conventions for the next thread

- Postgres 17, asyncpg, no migrations (CREATE IF NOT EXISTS, dev drops
  and recreates the database on schema change).
- Bring-your-own Postgres — no Docker compose in the repo. Host install,
  brew, or managed instance; the only contract is `DATABASE_URL`.
- Discord OAuth redirect_uri lives on the SPA — engine never knows the
  web URL.
- JWT bearer for everything (web, future bot, future extension). No
  cookie sessions.
- All `/api/*` mutation routes go through `Depends(require_user)` at
  router level, never per-route.
- Hooks (`Hook`, `EventHook`) emit events; the bus persists them; SSE
  fans out. Don't call event hooks from inside storage.
- One SQL file (`typoon/storage/schema.sql`) for the DDL; query
  methods in `typoon/storage/postgres.py` use string literals or
  small triple-quoted blocks.
