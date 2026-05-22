-- Typoon v3 schema — D1 / SQLite
--
-- Domain: translation tool with cross-device sync.
--
--   users               Discord-authenticated identity (+ tier from role)
--   identities          provider link (Discord OAuth)
--   api_tokens          programmatic access (Supporter tier+)
--
--   jobs                one translation run, ephemeral 7d
--   chapter_consumes    billing ledger (permanent)
--
--   reports             abuse moderation
--
-- 6 tables. Work-context cross-device sync via Workers KV
-- (binding WORK_CONTEXTS), not D1.
--
-- Apply order:
--   1. wrangler d1 execute typoon-db --remote --file=schema-drop.sql
--   2. wrangler d1 execute typoon-db --remote --file=schema.sql

-- ── Schema version ─────────────────────────────────────────────────

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
INSERT INTO meta (key, value) VALUES ('schema_version', '3');

-- ── Identity ───────────────────────────────────────────────────────

CREATE TABLE users (
  id                     INTEGER PRIMARY KEY,
  discord_id             TEXT    NOT NULL UNIQUE,
  display_name           TEXT    NOT NULL,
  avatar_url             TEXT,
  email                  TEXT,
  preferred_target_lang  TEXT,
  -- Tier driven by Discord role; see workers/api/src/lib/tiers.ts
  -- for quota config per tier. Cached here for queries; authoritative
  -- source is the role set fetched on each /auth/discord/exchange.
  tier_id                TEXT    NOT NULL DEFAULT 'free',
  tier_synced_at         TEXT,
  discord_roles          TEXT,             -- JSON array, debug snapshot
  created_at             TEXT    NOT NULL DEFAULT (datetime('now')),
  last_login_at          TEXT
);
CREATE INDEX idx_users_tier ON users(tier_id);

CREATE TABLE identities (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider    TEXT    NOT NULL,
  external_id TEXT    NOT NULL,
  metadata    TEXT,                          -- JSON, e.g. {"username":"…"}
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (provider, external_id)
);
CREATE INDEX idx_identities_user ON identities(user_id);

CREATE TABLE api_tokens (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name        TEXT    NOT NULL,
  token_hash  TEXT    NOT NULL UNIQUE,        -- SHA-256 hex
  prefix      TEXT    NOT NULL,               -- first 8 chars, for lookup
  scopes      TEXT    NOT NULL DEFAULT '[]',  -- JSON array
  last_used   TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  revoked_at  TEXT
);
CREATE INDEX idx_api_tokens_prefix ON api_tokens(prefix) WHERE revoked_at IS NULL;
CREATE INDEX idx_api_tokens_user   ON api_tokens(user_id);

-- ── Translation jobs ───────────────────────────────────────────────
--
-- One job = one chapter translation = one quota unit.
--
-- Lifecycle: init → uploading → pending → running → done | error
-- After 7 days from created_at, R2 lifecycle deletes raw/prepared/
-- archive blobs and a cleanup cron sets state='expired'.

CREATE TABLE jobs (
  id              INTEGER PRIMARY KEY,
  user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,

  -- Translation params
  source_lang     TEXT    NOT NULL,
  target_lang     TEXT    NOT NULL,
  -- Composite identifier from client (e.g. "mdex:abc-uuid", "upload:nano-id").
  -- When present, server hydrates context_in_key from KV ctx:{user_id}:{work_id}
  -- before pipeline starts; finalize callback writes merged context back.
  work_id         TEXT,
  -- 'translate' (full pipeline + archive) | 'analyze' (brief-only, no archive)
  kind            TEXT    NOT NULL DEFAULT 'translate'
                  CHECK (kind IN ('translate', 'analyze')),

  -- Lifecycle
  state           TEXT    NOT NULL DEFAULT 'init'
                  CHECK (state IN (
                    'init',      -- created, R2 multipart open
                    'uploading', -- parts being PUT to R2
                    'pending',   -- /start called, pipeline scheduled
                    'running',   -- pipeline active
                    'done',      -- archive ready
                    'error',     -- failed
                    'expired'    -- 7d cleanup, R2 blobs gone
                  )),
  progress_stage  TEXT,
  progress_index  INTEGER,
  progress_total  INTEGER,

  -- R2 keys
  zip_key         TEXT,                       -- raw/{id}/source.zip
  archive_key     TEXT,                       -- archive/{id}/output.bnl

  -- Sizing
  estimated_pages INTEGER,                    -- byte_size/1MB at init
  page_count      INTEGER,                    -- actual from prepare stage

  -- Error
  error_message   TEXT,

  -- Pipeline handles
  workflow_id     TEXT,                       -- Cloudflare Workflows instance

  -- WorkContext I/O (R2 keys, 7d lifecycle like everything else)
  context_in_key  TEXT,                       -- ctx/{id}/input.json.gz  (client supplied)
  context_out_key TEXT,                       -- ctx/{id}/output.json.gz (brief merged)

  -- Timestamps
  created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  started_at      TEXT,
  finished_at     TEXT,
  expires_at      TEXT    NOT NULL            -- created_at + 7d
);
CREATE INDEX idx_jobs_user_created ON jobs(user_id, created_at DESC);
CREATE INDEX idx_jobs_active       ON jobs(user_id, state)
  WHERE state IN ('init','uploading','pending','running');
CREATE INDEX idx_jobs_expires      ON jobs(expires_at)
  WHERE state != 'expired';
CREATE INDEX idx_jobs_user_work    ON jobs(user_id, work_id)
  WHERE work_id IS NOT NULL;

-- ── Billing ledger ─────────────────────────────────────────────────
--
-- Permanent. counted=1 → consumes monthly quota.
-- Reset = WHERE created_at >= date('now', 'start of month').

CREATE TABLE chapter_consumes (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  job_id      INTEGER NOT NULL REFERENCES jobs(id)  ON DELETE CASCADE,
  page_count  INTEGER NOT NULL,
  counted     INTEGER NOT NULL DEFAULT 1,    -- 1=quota; 0=waived
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_consumes_user_month
  ON chapter_consumes(user_id, created_at DESC)
  WHERE counted = 1;

-- Cross-device sync: handled by Workers KV (binding WORK_CONTEXTS for
-- per-work translation context; future bindings for library/settings as
-- needed). No D1 sync table — KV is cheaper and per-resource granular.

-- ── Moderation ─────────────────────────────────────────────────────

CREATE TABLE reports (
  id            INTEGER PRIMARY KEY,
  reporter_id   INTEGER REFERENCES users(id) ON DELETE SET NULL,
  job_id        INTEGER REFERENCES jobs(id)  ON DELETE SET NULL,
  reason        TEXT    NOT NULL,
  status        TEXT    NOT NULL DEFAULT 'open'
                CHECK (status IN ('open','reviewing','resolved','dismissed')),
  created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
  resolved_at   TEXT,
  resolver_id   INTEGER REFERENCES users(id) ON DELETE SET NULL,
  resolution    TEXT
);
CREATE INDEX idx_reports_status  ON reports(status, created_at DESC);
CREATE INDEX idx_reports_job     ON reports(job_id);
