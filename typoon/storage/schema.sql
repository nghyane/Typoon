-- Postgres 17 schema for Typoon.
--
-- Single source of truth, applied idempotently at PostgresStore.open().
-- No migrations in Phase 1: drop the database and re-create it when the
-- schema changes during development:
--   dropdb typoon && createdb -O typoon typoon
--
-- Conventions:
--   - BIGSERIAL primary keys.
--   - TIMESTAMPTZ NOT NULL DEFAULT NOW() for created_at / updated_at.
--   - JSONB for object/array payloads.
--   - tsvector generated columns for FTS — replaces FTS5 virtual tables
--     and their triggers. Tokenizer is `simple` (no stemming) because
--     stemming is meaningless for Vietnamese and harms proper-noun
--     precision.
--
--   - tasks.claim is one statement (FOR UPDATE SKIP LOCKED) — no
--     two-step pattern needed.

-- ── Schema version sentinel ─────────────────────────────────────────
-- Bump SCHEMA_VERSION in postgres.py when the DDL below changes shape.
-- Mismatch ⇒ refuse to start, instruct the operator to drop the volume.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- ── Identity ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL PRIMARY KEY,
    display_name  TEXT NOT NULL,
    avatar_url    TEXT,
    email         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS identities (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider     TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    metadata     JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(provider, external_id)
);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities(user_id);

-- ── Projects ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS projects (
    id           BIGSERIAL PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    description  TEXT,
    cover_path   TEXT,
    source_url   TEXT,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    owner_id     BIGINT REFERENCES users(id),
    shared       BOOLEAN NOT NULL DEFAULT FALSE,
    settings     JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_projects_owner_shared
    ON projects (owner_id) WHERE shared = TRUE;

CREATE TABLE IF NOT EXISTS chapters (
    id              BIGSERIAL PRIMARY KEY,
    project_id      BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx             DOUBLE PRECISION NOT NULL,
    title           TEXT,
    source_url      TEXT,
    rendered        BOOLEAN NOT NULL DEFAULT FALSE,
    page_count      INTEGER NOT NULL DEFAULT 0,
    -- Where the rendered archive lives (NULL until first render done).
    -- archive_backend is the artifact_store.backend_name; archive_locator
    -- is the opaque locator that store returned at upload time. URL build
    -- in the API dispatches by backend so multi-backend coexists without
    -- migration.
    archive_backend TEXT,
    archive_locator TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id, idx)
);

-- ── Project pins (per-user bookmarks) ───────────────────────────────

CREATE TABLE IF NOT EXISTS project_pins (
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    pinned_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, project_id)
);
CREATE INDEX IF NOT EXISTS idx_project_pins_user ON project_pins(user_id);

-- ── API tokens (long-lived auth for tools/extensions/CLI) ───────────

CREATE TABLE IF NOT EXISTS api_tokens (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,                  -- user-visible label
    token_hash  TEXT NOT NULL UNIQUE,           -- bcrypt(plaintext)
    prefix      TEXT NOT NULL,                  -- first 8 chars, shown in UI
    last_used   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_tokens_user_active
    ON api_tokens(user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_api_tokens_prefix_active
    ON api_tokens(prefix) WHERE revoked_at IS NULL;

-- ── Worker coordination ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS tasks (
    chapter_id   BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL CHECK(stage IN ('prepare','scan','translate','render')),
    claimed_by   TEXT,
    claimed_at   TIMESTAMPTZ,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    PRIMARY KEY (chapter_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(stage, claimed_by, claimed_at);

-- ── Bubbles + geometry ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bubbles (
    chapter_id   BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    bubble_idx   INTEGER NOT NULL,
    source_text  TEXT NOT NULL,
    confidence   REAL NOT NULL,
    shape_kind   TEXT NOT NULL DEFAULT 'dialogue'
        CHECK (shape_kind IN ('dialogue','burst')),
    source_text_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', source_text)) STORED,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);
CREATE INDEX IF NOT EXISTS idx_bubbles_source_text_tsv
    ON bubbles USING GIN (source_text_tsv);

CREATE TABLE IF NOT EXISTS bubble_geometry (
    chapter_id   BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    bubble_idx   INTEGER NOT NULL,
    polygon      JSONB NOT NULL,
    fit_box      JSONB NOT NULL,
    erase_box    JSONB NOT NULL,
    text_box     JSONB NOT NULL,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

CREATE TABLE IF NOT EXISTS page_geometry (
    chapter_id   BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    PRIMARY KEY (chapter_id, page_index)
);

-- ── Translations ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS translations (
    chapter_id      BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    translated_text TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK(kind IN ('dialogue','sfx','skip')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    translated_text_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', translated_text)) STORED,
    PRIMARY KEY (chapter_id, page_index, bubble_idx),
    FOREIGN KEY (chapter_id, page_index, bubble_idx)
        REFERENCES bubbles(chapter_id, page_index, bubble_idx) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_translations_text_tsv
    ON translations USING GIN (translated_text_tsv);

-- ── Chapter context briefs ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chapter_briefs (
    chapter_id   BIGINT PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    brief_json   JSONB NOT NULL,
    summary      TEXT,
    terms_text   TEXT,
    facts_text   TEXT,
    rules_text   TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_tsv   tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(summary,    '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(terms_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(facts_text, '')), 'C') ||
        setweight(to_tsvector('simple', coalesce(rules_text, '')), 'D')
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_chapter_briefs_search_tsv
    ON chapter_briefs USING GIN (search_tsv);

-- ── Project glossary ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS glossary (
    id           BIGSERIAL PRIMARY KEY,
    project_id   BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    notes        TEXT,
    source_term_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', source_term)) STORED,
    UNIQUE(project_id, source_term)
);
CREATE INDEX IF NOT EXISTS idx_glossary_source_term_tsv
    ON glossary USING GIN (source_term_tsv);

-- ── Events (event bus persistence + replay) ─────────────────────────

CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_events_chapter_pagedone
    ON events ((data->>'chapter_id'))
    WHERE data->>'type' = 'PageDone';

-- ── updated_at maintenance ──────────────────────────────────────────
-- Postgres has no "ON UPDATE" trigger sugar — keep the explicit triggers.

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chapters_touch_updated_at ON chapters;
CREATE TRIGGER chapters_touch_updated_at
    BEFORE UPDATE ON chapters
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS projects_touch_updated_at ON projects;
CREATE TRIGGER projects_touch_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

-- Cascade chapter writes to project.updated_at.

CREATE OR REPLACE FUNCTION touch_project_via_chapter() RETURNS TRIGGER AS $$
BEGIN
    UPDATE projects SET updated_at = NOW()
        WHERE id = COALESCE(NEW.project_id, OLD.project_id);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chapters_touch_project ON chapters;
CREATE TRIGGER chapters_touch_project
    AFTER INSERT OR UPDATE OR DELETE ON chapters
    FOR EACH ROW
    EXECUTE FUNCTION touch_project_via_chapter();

-- Task lifecycle bumps chapter.updated_at — UI uses that as freshness.

CREATE OR REPLACE FUNCTION touch_chapter_via_task() RETURNS TRIGGER AS $$
BEGIN
    UPDATE chapters SET updated_at = NOW()
        WHERE id = COALESCE(NEW.chapter_id, OLD.chapter_id);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tasks_touch_chapter ON tasks;
CREATE TRIGGER tasks_touch_chapter
    AFTER INSERT OR UPDATE OR DELETE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION touch_chapter_via_task();
