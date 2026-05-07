# Phase 1 — handoff state

This file is a snapshot of where Typoon stands as we hand off to a
new thread that will implement RFC-005 (Postgres-only). Read this
before opening the codebase.

## Where we are

- Single-process self-host on a Mac. SQLite at `~/.typoon/typoon.db`,
  artifacts at `~/.typoon/artifacts/`, projects at `~/.typoon/projects/`.
- Engine API: `typoon api` on `:8000`.
- Workers: `typoon work --role full` in the same process group, polling
  the SQLite tasks table.
- Web: Vite dev on `:5173`, proxies `/api` and `/files` to `:8000`.
- Auth: Discord OAuth, JWT bearer, `users` + `identities` tables. SPA
  owns the OAuth redirect. Backend is a pure JSON exchanger.

## What works (verified)

- Project CRUD: create empty project + cover upload + delete.
- Chapter ingestion: PDF, CBZ/ZIP, multi-image upload via
  `POST /api/projects/{id}/chapters/upload`.
- Pipeline: prepare → scan → translate → render runs end-to-end.
- Bubble OCR list, manual translation edit (PATCH).
- Glossary CRUD with FTS5 search.
- Project settings (title, description, target_lang).
- Chapter redo + delete.
- SSE event stream with disconnect cleanup.
- Workers dashboard (queue depth per stage).
- FTS search (LLM agent's `search_knowledge` tool — important for
  translation context).
- Discord OAuth login (SPA-driven flow):
  - `GET /api/auth/config` — public client_id + guild branding (name, icon, invite). 503 if Discord Server Widget not enabled and gating is on.
  - `POST /api/auth/discord/exchange { code, redirect_uri }` — used by both web standalone and Discord Activity.
  - `GET /api/auth/me` — current user + guild branding.
  - `POST /api/auth/logout` — no-op stub for stateful logout when added.
  - SPA owns redirect_uri (`window.origin + /auth/callback`) and CSRF state (sessionStorage). No cookies, no engine HTML.
- Guild branding: name from Discord widget endpoint, icon hash cached from `/users/@me/guilds` on first member login. Engine fails loud (503) instead of silently substituting a placeholder name.
- Bootstrap admin via `TYPOON_BOOTSTRAP_DISCORD_ID` env.
- Guild gating via `DISCORD_GUILD_ID` env.

## What does not work yet

- Multi-host: SQLite cannot serve API on a VPS + worker on a Mac. This
  is the main motivator for RFC-005.
- Postgres event bus: `PostgresEventBus` exists but has never been
  exercised against a real Postgres. It needs a smoke test.
- Object storage: artifacts only on the same disk as the API.

## Code map

```
typoon/
  adapters/          external systems
    artifact_store.py    LocalArtifactStore
    chapter_archive.py   prepared_key/render_key/masks_key + pack_and_upload
    event_bus.py         InProcessEventBus + PostgresEventBus + factory
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
    sqlite.py          SqliteStore — to be replaced (RFC-005)
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
```

All `/api/*` except `/api/auth/{config,discord/exchange}` and the SSE
stream require `Authorization: Bearer <jwt>`. SSE accepts the token via
`?token=` because `EventSource` cannot set headers.

## Local setup

```bash
# 1. Postgres (only the database for now; API + worker run on host)
docker compose up -d db

# 2. Engine config — copy .env.example to .env and fill in:
#    DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, DISCORD_GUILD_ID,
#    TYPOON_BOOTSTRAP_DISCORD_ID
#    DATABASE_URL=postgresql://typoon:typoon@localhost:5432/typoon
cp .env.example .env

# 3. Engine
typoon api --port 8000      # one terminal
typoon work --role full     # another terminal

# 4. Web
cd web && npm install && npm run dev
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

The **only** thing required for the next thread to start is RFC-005:
Postgres-only storage cutover. Everything else listed below is
backlog.

Backlog (loosely ordered):

1. **RFC-005** — Postgres-only (in `docs/rfc/005-postgres-only.md`).
   17 ordered steps documented inline.
2. Discord Activity SDK integration (Phase 2).
3. Discord bot service (separate repo, consumes API).
4. Browser extension for quick-import (separate repo).
5. Object storage (R2) — only when API and worker are on different
   physical hosts.
6. Production Dockerfile for the API container.
7. nginx reverse proxy config.
8. Tailscale setup guide for worker ↔ API ↔ DB.
9. CORS strict mode + healthz endpoint.
10. Bubble editor UI.
11. Chapter viewer (full-size manga page reader).
12. Search palette (⌘K) wired to `/api/search`.
13. Bot-issued per-user API tokens (when bot lands).
14. Quota enforcement (when payment lands).

## Conventions for the next thread

- Postgres 17, asyncpg, no migrations (CREATE IF NOT EXISTS, dev nukes
  the volume on schema change).
- Single `compose.yml` at repo root, no dev/prod split.
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
