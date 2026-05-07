# RFC-005: Postgres-only storage + closed multi-host topology

Status: **proposed** — implement on a separate branch
Owner: next agent / contributor

## Summary

Replace SQLite with Postgres 17 as the only supported database. Drop the
in-process event bus along with it. The engine becomes a pair of
processes — a stateless API container on the host that owns the public
domain, and one or more workers that connect to the same Postgres over
a private network (Tailscale). Schema is recreated on every code
change; there are no migrations in this phase.

## Why

1. **Multi-host is the target topology**: API on a VPS, workers on a Mac
   with Metal GPU. SQLite cannot serve two physical hosts.
2. **The current dual-mode logic is leaky**: `is_postgres()` branches in
   `make_event_bus`, `deps.get_store`, `app.py` role check, and the
   workers loop. Every new feature has to remember to handle both. Code
   is full of "if Postgres, raise NotImplementedError" placeholders that
   are now blocking.
3. **FTS quality matters**: the LLM agent's `search_knowledge` tool
   relies on `glossary_search`, `search_briefs`, and `search_context`.
   Falling back to LIKE degrades translation quality. Postgres tsvector
   is strictly better than SQLite FTS5 for our use case (Vietnamese
   tokenization, ranking, generated columns instead of triggers).
4. **The schema is small enough that no migration tool is needed**.
   Phase 1 community is closed and small; if the schema changes during
   development we drop the database and let `CREATE TABLE IF NOT EXISTS`
   re-run. Alembic comes when paying users have data we cannot lose.

## Non-goals (Phase 1)

- No SQLite anywhere — not even for tests. Tests spin up a throwaway
  Postgres database via the same compose file.
- No migration tooling (Alembic, Flyway). Schema lives in one
  `schema.sql` file applied idempotently at boot.
- No object storage. Workers and API share `~/.typoon/artifacts/` for
  now. Phase 2 introduces R2 / HTTP relay when a real VPS exists.
- No Discord Activity SDK. Phase 1 ships web standalone over Discord
  OAuth. DA is a second entry point added in Phase 2 without changing
  the engine.

## Architecture

```
┌──────────── Tailnet ────────────┐
│                                 │
│  Mac (laptop)        VPS host   │
│  ───────────         ─────────  │
│  typoon work     ←→  postgres   │  ← single source of truth
│                  ←→  typoon api │  ← stateless, scales horizontally
│                                 │     when needed
└─────────────────────────────────┘
                                  ↑
                                  │ HTTPS public
                              Web users
```

Phase 1 (dev): everything on the same Mac. Compose runs Postgres only;
the API runs on the host (`typoon api`) for hot-reload, the worker
runs on the host (`typoon work`) for GPU access, the web dev server
runs on the host (`vite`).

Phase 2 (prod): same code, just split deployment. API + Postgres in
Docker on the VPS, worker on the Mac, both joined by Tailscale.

## Code changes

### Storage layer

- **Add** `typoon/storage/postgres.py` — `PostgresStore`. Same `Store`
  protocol as today. asyncpg pool (min=2, max=10), one connection per
  call, no shared cursors.
- **Add** `typoon/storage/schema.sql` — full DDL in one file. Run via
  `pool.execute(open(schema_sql).read())` at `PostgresStore.open()`.
  Idempotent: every CREATE uses `IF NOT EXISTS`.
- **Delete** `typoon/storage/sqlite.py`.
- **Update** `typoon/storage/__init__.py` to export `PostgresStore`
  instead of `SqliteStore`. Keep the `Store` protocol unchanged.

### Schema notes (per table)

Same logical schema as today. Translation rules:

| Concept | SQLite | Postgres |
|---|---|---|
| Auto-increment PK | `INTEGER PRIMARY KEY` | `BIGSERIAL PRIMARY KEY` (or `GENERATED ALWAYS AS IDENTITY`) |
| Timestamp default | `TEXT NOT NULL DEFAULT (datetime('now'))` | `TIMESTAMPTZ NOT NULL DEFAULT NOW()` |
| Boolean | `INTEGER NOT NULL DEFAULT 0` | `BOOLEAN NOT NULL DEFAULT FALSE` |
| JSON blob | `TEXT` | `JSONB` |
| FTS storage | virtual table `xxx_fts USING fts5` + triggers | generated tsvector column + GIN index |
| FTS trigger | `bubbles_ai`, `_ad`, `_au` per table | not needed (generated column auto-maintains) |
| Updated_at touch | trigger that overwrites the column | trigger or column-level `ON UPDATE` (Postgres has no native, keep trigger) |
| Conflict upsert | `INSERT OR REPLACE` | `INSERT ... ON CONFLICT (...) DO UPDATE SET ...` |
| Locking row claim | two-step SELECT + UPDATE with race recheck | `SELECT ... FOR UPDATE SKIP LOCKED` (atomic, no recheck needed) |

The `tasks` claim function is the most important to get right: today it
does a two-step pattern because aiosqlite cannot return a row from
UPDATE atomically. With Postgres we can do this in one statement:

```sql
UPDATE tasks
SET    claimed_by = $1, claimed_at = NOW()
WHERE  ctid = (
    SELECT ctid FROM tasks
    WHERE  stage = $2
      AND  (claimed_by IS NULL OR claimed_at < NOW() - INTERVAL '10 minutes')
    ORDER  BY chapter_id
    FOR UPDATE SKIP LOCKED
    LIMIT  1
)
RETURNING chapter_id;
```

That replaces the 35-line two-step in `SqliteStore.claim_task`.

### FTS via tsvector

Replace four FTS5 virtual tables and ~12 triggers with four generated
columns and four GIN indexes:

```sql
ALTER TABLE bubbles
  ADD COLUMN source_text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', source_text)) STORED;
CREATE INDEX bubbles_source_text_tsv ON bubbles USING GIN (source_text_tsv);

ALTER TABLE translations
  ADD COLUMN translated_text_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', translated_text)) STORED;
CREATE INDEX translations_translated_text_tsv ON translations USING GIN (translated_text_tsv);

ALTER TABLE chapter_briefs
  ADD COLUMN search_tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('simple', coalesce(summary,    '')), 'A') ||
    setweight(to_tsvector('simple', coalesce(terms_text, '')), 'B') ||
    setweight(to_tsvector('simple', coalesce(facts_text, '')), 'C') ||
    setweight(to_tsvector('simple', coalesce(rules_text, '')), 'D')
  ) STORED;
CREATE INDEX chapter_briefs_search_tsv ON chapter_briefs USING GIN (search_tsv);

ALTER TABLE glossary
  ADD COLUMN source_term_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', source_term)) STORED;
CREATE INDEX glossary_source_term_tsv ON glossary USING GIN (source_term_tsv);
```

Tokenizer choice: `simple` lowercases and splits on whitespace, no
stemming. That matches Vietnamese (where stemming is meaningless) and
keeps proper nouns intact. Do not use `english` — it stems "magic" and
"magical" to the same token, which loses precision for translation
context.

Query API: use `websearch_to_tsquery` so the agent can pass natural
strings (`"phép thuật"`, `magic OR sorcery`, `-cấm`) without the engine
sanitising. Rank with `ts_rank`, order DESC, limit. Caller wraps in the
same `[Ch{idx} pX] source → translated` text format that already exists.

### Event bus

- **Delete** `InProcessEventBus` and `make_event_bus()` factory.
- **Rename** `PostgresEventBus` to just `EventBus` (concrete class).
  The Protocol stays as `EventBus` for type hints — keep both names by
  making the class implement the protocol implicitly. Suggestion:
  rename Protocol to `EventBusProtocol` to remove the ambiguity.
- **Smoke-test** the existing `LISTEN/NOTIFY` code with a real Postgres
  instance. The current code was written six weeks ago and never run
  end-to-end against a server — only against the test client. Pay
  particular attention to:
  - Connection lost recovery (Postgres restart, network blip)
  - Missed events when the listener reconnects (replay via `events.id > last_id`)
  - Multiple workers each holding their own LISTEN connection (no
    shared connection)

### Configuration

Add a `[server]` section to `config.toml`:

```toml
[server]
public_api_url = "http://localhost:8000"
public_web_url = "http://localhost:5173"
host = "0.0.0.0"
port = 8000

[auth]
discord_client_id     = ""
# discord_client_secret comes from env DISCORD_CLIENT_SECRET
discord_guild_id      = ""
bootstrap_discord_id  = ""
# discord_redirect_uri = derived from server.public_api_url
```

`load_config()`:

- Derives `auth.discord_redirect_uri = f"{server.public_api_url}/api/auth/discord/callback"`.
  Drop the manual override field.
- Honors `DATABASE_URL` env first, then `database_url` toml. **Errors
  out if the value is empty or sqlite-shaped** — there is no fallback.
- Stores `auth.web_url = server.public_web_url` for the OAuth bootstrap
  HTML to redirect to after stashing the token.
- Reads `host` / `port` for the CLI `typoon api` command (currently
  hardcoded `0.0.0.0:8000`).

### Auth route

The bootstrap HTML in `routes/auth.py` currently does
`window.location.replace("/")`. Change to
`window.location.replace("{web_url}")` injected from the resolved
config. No more 404 when API and web are on different ports.

CORS: `allow_origins=["*"]` is wrong for a closed community. Tighten to
`[server.public_web_url]` plus `https://discord.com`,
`https://*.discordsays.com` (for Phase 2 DA).

State cookie: `secure=server.public_api_url.startswith("https")`. Dev
HTTP-only OK; prod must be HTTPS so the Secure flag flips on.

### Removed types and call sites

After the cutover, grep the repo for these and remove:

- `aiosqlite` (top-level import)
- `SqliteStore`, `SqliteStore.open`, `SqliteStore.open_memory`
- `InProcessEventBus`
- `make_event_bus`
- `is_postgres`
- `paths.db` (the SQLite file path) — keep `paths.root`, drop `paths.db`
- `_migrate_chapters`, `_migrate_projects` (only existed because DB
  shipped before columns were finalised; new schema.sql has them all)
- The `database_url: str = ""` default in `Config` — make it required

Files definitely affected (count from grep):

- `typoon/api/app.py` — drop `is_postgres` role check
- `typoon/api/deps.py` — drop branch
- `typoon/adapters/event_bus.py` — slim to just one class + the `EventHook`
- `typoon/adapters/projects.py` — drop `Projects.open`'s SQLite path,
  rewire to `PostgresStore.open(config.database_url)`
- `typoon/workers/loop.py` — drop `_validate_role_vs_db`
  (Postgres + role=full or role=vision/llm/api are all fine), drop
  `is_postgres` import
- `typoon/storage/__init__.py` — switch export
- `typoon/storage/store.py` — keep, no changes
- `typoon/paths.py` — drop `db` property

### Workers

`typoon work --role full` continues to work in Phase 1 (dev all on
Mac). Once Postgres is the only backend, `--role api`, `--role vision`,
`--role llm` work too — current code already supports them, only the
SQLite role guard was blocking. Drop `_validate_role_vs_db`.

Worker connects via `DATABASE_URL` env (Tailscale-resolvable host in
prod). Same env for API.

## Compose

Already added in this PR:

```yaml
# compose.yml
services:
  db:
    image: postgres:17-alpine
    ports: ["${POSTGRES_PORT:-5432}:5432"]
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck: pg_isready
    environment: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
volumes:
  pgdata:
```

`docker compose up -d db` is the only setup step. The host runs the
API and worker.

When the VPS exists, the same compose file gains an `api` service and
an `nginx` reverse proxy. Nothing about Phase 1 forces dev/prod splits.

## Schema dropped on schema change

This is dev-only behaviour and intentional:

```bash
docker compose down -v   # nuke pgdata
docker compose up -d db  # recreate fresh
```

API on next boot runs `schema.sql`, ends up with the new shape. No
migration scripts. When real users have data this rule changes.

## Order of work

Implement in this order. Each step compiles, type-checks, tests pass.
Do not commit a step that breaks the previous one.

1. **`schema.sql` complete + `PostgresStore` skeleton (open/close/pool)**
2. **Port `users`, `identities`, `projects`, `chapters` queries**
3. **Port `tasks` and `claim_task` (`FOR UPDATE SKIP LOCKED`)**
4. **Port `bubbles`, `bubble_geometry`, `page_geometry`, `translations`**
5. **Port `glossary`, `chapter_briefs`**
6. **Port FTS queries — `glossary_search`, `search_briefs`, `search_context`**
7. **Port `queue_stats`, `update_translation`, `delete_chapter`,
   `delete_chapter_data`, `delete_project`, the rest**
8. **Port `updated_at` triggers (chapters, projects)**
9. **Port `append_event` / `get_events_after` (event bus persistence)**
10. **Wire `PostgresStore` into `deps.py` and `Projects.open`**
11. **Drop `SqliteStore`, `InProcessEventBus`, `is_postgres`, dead
    helpers; clean imports**
12. **Refactor `Config` for `[server]` block + URL derivation**
13. **Auth bootstrap HTML uses `web_url` from config**
14. **CORS strict, cookie secure derived**
15. **Add `/api/healthz` (returns 200 + DB ping)**
16. **Live test: `docker compose up`, `typoon api`, `typoon work`,
    `npm run dev`. Run the full e2e: login, create project, upload
    chapter, watch worker process it through SSE.**
17. **Update README with the new dev setup**

## Risks

- **Concurrency parity with SQLite WAL**. SQLite WAL is forgiving;
  Postgres surfaces deadlocks, serialization failures. Test the worker
  loop under load (10+ concurrent claim_task) before declaring done.
- **JSON column changes**. Today `chapter_briefs.brief_json` is `TEXT`
  with manual `json.loads`. Postgres `JSONB` is faster and queryable
  but `asyncpg` returns Python dicts directly — adjust call sites that
  expect to call `json.loads` themselves.
- **Datetime handling**. SQLite returns `'2026-05-07 09:43:03'` strings;
  Postgres returns `datetime` objects. The API serializes dates as ISO
  strings; ensure `ProjectOut.from_row` and friends handle both
  explicitly (or pick one and stick to it). Recommend: convert all
  timestamps to ISO 8601 string at the SQL layer with `TO_CHAR(...)` to
  match the API contract — no breaking change to clients.
- **FTS query syntax differences**. `websearch_to_tsquery` accepts
  Google-style; the LLM tool currently passes raw strings. Test that
  agent queries like `"Solo Leveling"` (with quotes) work as phrase
  matches. If the agent's input contains punctuation that breaks
  Postgres parser, sanitize at the storage boundary.
- **EventBus race during cutover**. If anyone runs the new code against
  a Postgres that still has stale rows in the events table, sequence
  numbers may collide. Add a startup check that errors out if the
  schema version (a row in a `meta` table, single key/value) doesn't
  match what the code expects, and instructs the operator to nuke the
  volume.

## Things explicitly out of scope

- Any reference to R2, S3, MinIO, object storage in general
- Discord Activity SDK
- Bot service, browser extension
- nginx, Dockerfile for the API container, CI
- Telemetry, structured logging, request IDs

These belong to later RFCs.

## Open questions for the implementer

1. Should `claim_task` run in a stored procedure or stay as application
   SQL? Stored proc is faster (one round trip) and atomic, but harder
   to inspect. Recommend: keep as application SQL with `RETURNING`,
   revisit if profile shows it as a bottleneck.
2. Use `JSONB` for `chapter_briefs.brief_json` and `identities.metadata`,
   or stick with TEXT? Recommend: JSONB. Migration cost is zero (we
   nuke schema on change), and we get GIN-indexable JSON for free.
3. Should the events table grow unbounded? Today no cleanup. Add a
   nightly `DELETE FROM events WHERE created_at < NOW() - INTERVAL '7 days'`?
   Recommend: yes, but add it as an admin endpoint, not auto. Phase 1
   community is small enough that 7-day events fit in MB.
