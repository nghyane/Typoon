-- Postgres 17 schema for Typoon — v5 Material + Translation architecture.
--
-- See docs/rfc/material-architecture.md for the design rationale. This
-- file is the source of truth; bump SCHEMA_VERSION in postgres.py and
-- drop-recreate the DB whenever shape changes. No migration tooling.
--
-- Layering (top-down):
--   Identity                users, identities, user_guilds, api_tokens
--   Reading entity          materials, chapters, material_link_votes
--   Pipeline (chapter)      bubbles, bubble_geometry, page_geometry
--   Pipeline (lang+gloss)   translation_drafts, translation_draft_bubbles,
--                           draft_briefs
--   Pipeline (user)         translations, translation_edits
--   Library (per-user)      library_entries, library_materials
--   Glossary                user_glossary, community_glossary
--   Translator memory       translator_memory, translator_memory_briefs
--   Queue                   tasks
--   Quota / Moderation      chapter_consumes, reports, moderation_actions
--   Inbox (browser upload)  material_inbox
--
-- Conventions:
--   - BIGSERIAL primary keys; foreign keys ON DELETE CASCADE when the
--     row is meaningless without its parent, SET NULL when the row is
--     audit-bearing (history kept).
--   - TIMESTAMPTZ NOT NULL DEFAULT NOW() for created_at / updated_at.
--   - JSONB for object payloads; TEXT[] for short tag-like lists.
--   - tsvector generated columns for FTS, tokenizer 'simple' (no
--     stemming — Vietnamese + proper nouns).
--   - tasks.claim is one statement (FOR UPDATE SKIP LOCKED). No two-step.

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

-- Cached Discord guild memberships. Refreshed at login + lazily on
-- spawn/feed access. Drives the `scope_guild_id` resolution for
-- translation drafts and the `/api/feed/guild/{id}` membership check.

CREATE TABLE IF NOT EXISTS user_guilds (
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    guild_id     TEXT NOT NULL,                 -- Discord snowflake
    guild_name   TEXT,
    guild_icon   TEXT,                          -- discord cdn url
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, guild_id)
);
CREATE INDEX IF NOT EXISTS idx_user_guilds_guild ON user_guilds(guild_id);

-- API tokens (CLI / extension / worker). Web SPA uses Discord JWT, not tokens.

CREATE TABLE IF NOT EXISTS api_tokens (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,                  -- user-visible label
    token_hash  TEXT NOT NULL UNIQUE,           -- bcrypt(plaintext)
    prefix      TEXT NOT NULL,                  -- first 8 chars, shown in UI
    -- Empty = ordinary client token (read-only on user's own data).
    -- 'worker' grants /api/blobs/* for pipeline traffic.
    scopes      TEXT[] NOT NULL DEFAULT '{}',
    last_used   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_tokens_user_active
    ON api_tokens(user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_api_tokens_prefix_active
    ON api_tokens(prefix) WHERE revoked_at IS NULL;

-- ── Material ────────────────────────────────────────────────────────
-- A manga identity. Source-backed materials are cross-user (unique on
-- (source, upstream_ref)); two users importing the same HappyMH manga
-- share this row. Extension / upload materials are per-row (source IS
-- NULL → no dedup).
--
-- `imported_by` is audit only — the first user who triggered the row
-- creation. It is NOT an ownership boundary; subsequent users have
-- the same read/spawn capabilities.

CREATE TABLE IF NOT EXISTS materials (
    id            BIGSERIAL PRIMARY KEY,
    imported_by   BIGINT REFERENCES users(id) ON DELETE SET NULL,
    origin        TEXT NOT NULL CHECK (origin IN ('source','extension','upload')),

    -- Source-backed identity. NULL for ext + upload.
    source        TEXT,
    upstream_ref  TEXT,

    -- Display snapshot. First-write wins; later refreshes happen out of
    -- band when the manifest reports a change.
    title         TEXT NOT NULL,
    cover_url     TEXT,
    description   TEXT,
    author        TEXT,
    status        TEXT,
    languages     TEXT[] NOT NULL DEFAULT '{}',

    -- Identity hints used by the community-voted suggestion ranker.
    -- Manifest populates what the source exposes; left NULL otherwise.
    -- See features/browse/manifest/types.ts for the selector slots.
    title_native  TEXT,
    title_alt     TEXT[],
    cross_refs    JSONB,           -- {"mdex_uuid":"…","anilist":12345}

    -- NSFW gate. Set by manifest.nsfw, by ext UI, or by user during
    -- upload. Forces draft visibility='private' regardless of opt-out.
    nsfw          BOOLEAN NOT NULL DEFAULT FALSE,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source-backed dedup. Constraint applies only when source IS NOT NULL
-- so ext / upload materials remain per-row.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_materials_source_ref
    ON materials (source, upstream_ref)
    WHERE source IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_materials_imported_by ON materials(imported_by);
CREATE INDEX IF NOT EXISTS idx_materials_title_native
    ON materials(title_native) WHERE title_native IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_materials_cross_refs
    ON materials USING GIN (cross_refs)
    WHERE cross_refs IS NOT NULL;

-- ── Material link votes (cross-source identity, community-driven) ──
-- One row per (user, pair). Canonical ordering on the pair
-- (material_a_id < material_b_id) avoids storing A↔B twice. Aggregated
-- via the materialized view below; refresh nightly is enough — vote
-- changes propagate to suggestions lazily, which is fine for UX.

CREATE TABLE IF NOT EXISTS material_link_votes (
    material_a_id  BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    material_b_id  BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    voter_id       BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vote           SMALLINT NOT NULL CHECK (vote IN (-1, 1)),
    voted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (material_a_id, material_b_id, voter_id),
    CHECK (material_a_id < material_b_id)
);
CREATE INDEX IF NOT EXISTS idx_link_votes_a ON material_link_votes(material_a_id);
CREATE INDEX IF NOT EXISTS idx_link_votes_b ON material_link_votes(material_b_id);

-- Aggregate score view. Refreshed by a cron / on-demand from the
-- suggestion endpoint — staleness is acceptable.
CREATE MATERIALIZED VIEW IF NOT EXISTS material_links AS
SELECT material_a_id,
       material_b_id,
       SUM(vote)::INTEGER  AS score,
       COUNT(*)::INTEGER   AS total_votes
FROM material_link_votes
GROUP BY material_a_id, material_b_id;
CREATE UNIQUE INDEX IF NOT EXISTS uniq_material_links
    ON material_links(material_a_id, material_b_id);
CREATE INDEX IF NOT EXISTS idx_links_a ON material_links(material_a_id);
CREATE INDEX IF NOT EXISTS idx_links_b ON material_links(material_b_id);

-- ── Chapter ─────────────────────────────────────────────────────────
-- A unit of pages inside a Material. Pages are always local: the
-- prepare stage fetches + stores them in our blob store as prepared.bnl.
--
-- `prepared_hash` is the content-addressable cache key. Two chapter
-- rows with identical pixel content (same upload zip, identical
-- source content) share the same prepared.bnl blob via this hash.
-- Downstream caches (bubbles, drafts) key off chapter_id which is
-- still per-row, but the expensive scan/translate runs that follow
-- a hash collision can be reused — see translation_drafts.

CREATE TABLE IF NOT EXISTS chapters (
    id                BIGSERIAL PRIMARY KEY,
    material_id       BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,

    -- Sparse server-managed sort key (see _resolve_chapter_position in
    -- postgres.py). Same INITIAL_GAP / REBALANCE_MIN_GAP policy as the
    -- old projects schema.
    position          INTEGER NOT NULL,

    -- Display string. Free-form: "4", "4.5", "Extra", "Oneshot",
    -- "v2 ch.1", "". Not unique within a material.
    number            TEXT NOT NULL,
    label             TEXT,                 -- full label as the source presents
    upstream_url      TEXT,                 -- chapter URL on source; NULL for upload

    -- CAS for prepared.bnl. SHA256 hex string; NULL until prepare runs.
    prepared_hash     TEXT,
    prepared_backend  TEXT,
    prepared_locator  TEXT,

    -- Masks output of the scan stage. Per-chapter (geometry only
    -- depends on pixel content, not lang/glossary).
    masks_backend     TEXT,
    masks_locator     TEXT,

    page_count        INTEGER NOT NULL DEFAULT 0,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (material_id, position)
);
CREATE INDEX IF NOT EXISTS idx_chapters_material
    ON chapters(material_id, position);
CREATE INDEX IF NOT EXISTS idx_chapters_prepared_hash
    ON chapters(prepared_hash) WHERE prepared_hash IS NOT NULL;

-- ── Chapter inbox (browser-direct upload handle) ────────────────────
-- Persists multipart upload coordinates between `/upload-finalize`
-- (returns 202 immediately) and the prepare worker. One row per
-- chapter; deleted by the worker after prepare succeeds + the inbox
-- key is removed from R2.

CREATE TABLE IF NOT EXISTS material_inbox (
    chapter_id   BIGINT PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    tmp_id       TEXT NOT NULL,
    upload_id    TEXT NOT NULL,
    parts        JSONB NOT NULL,           -- [{"number": 1, "etag": "..."}]
    title        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Scan output — chapter level (Layer 1) ───────────────────────────
-- Bubbles, geometry, and masks bind to chapter_id, not translation.
-- They depend only on pixel content, so every translation on this
-- chapter (any lang, any glossary) reads the same data.

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

-- ── Translation drafts — lang + glossary level (Layer 2) ────────────
-- LLM output keyed on (chapter, source_lang, target_lang, glossary_fp).
-- Visibility controls who can reuse the draft as cache. The unique
-- index excludes 'private' drafts so the same user/key can spawn a
-- fresh private draft alongside a guild-shared one.
--
-- DMCA: when `takedown_at` is set, the draft is invisible to readers
-- and cache lookup; any translation referencing it shows a takedown
-- placeholder until owner re-spawns.

CREATE TABLE IF NOT EXISTS translation_drafts (
    id                 BIGSERIAL PRIMARY KEY,
    chapter_id         BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    source_lang        TEXT NOT NULL,
    target_lang        TEXT NOT NULL,
    glossary_fp        TEXT NOT NULL,           -- 16-hex SHA256 of applied glossary

    llm_model          TEXT NOT NULL,
    created_by         BIGINT REFERENCES users(id) ON DELETE SET NULL,

    visibility         TEXT NOT NULL DEFAULT 'guild'
        CHECK (visibility IN ('private','guild','all_guilds')),
    -- The guild this draft was spawned from. NULL when visibility is
    -- 'all_guilds' (creator publishes across every guild they belong
    -- to) or 'private' (no guild scope).
    scope_guild_id     TEXT,

    -- DMCA. Set by admin takedown; cascades visibility off.
    takedown_at        TIMESTAMPTZ,
    takedown_reason    TEXT,

    state              TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending','running','done','error')),
    error_message      TEXT,
    progress_stage     TEXT,
    progress_index     INTEGER,
    progress_total     INTEGER,

    -- Default render archive. Every translation pointing at this draft
    -- with no per-translation override (sparse edits) serves from here.
    archive_backend    TEXT,
    archive_locator    TEXT,
    rendered_at        TIMESTAMPTZ,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Cache key uniqueness. Excludes private + taken-down drafts so a
-- private spawn can coexist and a takedown can be replaced.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_drafts_cache
    ON translation_drafts (chapter_id, source_lang, target_lang, glossary_fp)
    WHERE visibility != 'private' AND takedown_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_drafts_creator ON translation_drafts(created_by);
CREATE INDEX IF NOT EXISTS idx_drafts_chapter ON translation_drafts(chapter_id);

CREATE TABLE IF NOT EXISTS translation_draft_bubbles (
    draft_id        BIGINT NOT NULL REFERENCES translation_drafts(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    translated_text TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('dialogue','sfx','skip')),
    translated_text_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', translated_text)) STORED,
    PRIMARY KEY (draft_id, page_index, bubble_idx)
);
CREATE INDEX IF NOT EXISTS idx_draft_bubbles_text_tsv
    ON translation_draft_bubbles USING GIN (translated_text_tsv);

-- LLM context brief — per draft (each run has its own glossary +
-- target lang context, so briefs don't reuse across drafts).
CREATE TABLE IF NOT EXISTS draft_briefs (
    draft_id    BIGINT PRIMARY KEY REFERENCES translation_drafts(id) ON DELETE CASCADE,
    brief_json  JSONB NOT NULL,
    summary     TEXT,
    terms_text  TEXT,
    facts_text  TEXT,
    rules_text  TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_tsv  tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(summary,    '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(terms_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(facts_text, '')), 'C') ||
        setweight(to_tsvector('simple', coalesce(rules_text, '')), 'D')
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_draft_briefs_search_tsv
    ON draft_briefs USING GIN (search_tsv);

-- ── Translation — per-user wrapper (Layer 3) ────────────────────────
-- One row per (chapter, owner, target_lang). Points at a draft (Layer
-- 2) and optionally carries sparse edits + a per-user render archive
-- (when the user's edits diverge from the draft's default render).
--
-- `in_feed` controls Hội Mê Truyện feed inclusion — independent from
-- draft visibility: a private draft can yield a non-feed translation;
-- a guild-shared draft can be excluded from feed by its owner.

CREATE TABLE IF NOT EXISTS translations (
    id                 BIGSERIAL PRIMARY KEY,
    chapter_id         BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    owner_id           BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_lang        TEXT NOT NULL,
    draft_id           BIGINT REFERENCES translation_drafts(id) ON DELETE SET NULL,

    -- Render archive. NULL means "fall back to draft's default render"
    -- — used when the user has no edits and shares the draft's archive.
    archive_backend    TEXT,
    archive_locator    TEXT,
    rendered_at        TIMESTAMPTZ,

    -- Feed flag — guild-scoped via feed_guild_id. NULL feed_guild_id
    -- with in_feed=TRUE means "publish in every guild owner belongs to".
    in_feed            BOOLEAN NOT NULL DEFAULT TRUE,
    feed_guild_id      TEXT,

    takedown_at        TIMESTAMPTZ,
    takedown_reason    TEXT,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (chapter_id, owner_id, target_lang)
);
CREATE INDEX IF NOT EXISTS idx_translations_owner ON translations(owner_id);
CREATE INDEX IF NOT EXISTS idx_translations_chapter ON translations(chapter_id);
CREATE INDEX IF NOT EXISTS idx_translations_feed
    ON translations(feed_guild_id, created_at DESC)
    WHERE in_feed = TRUE AND takedown_at IS NULL;

-- Sparse edits over the shared draft. Reader loads draft bubbles +
-- overlays edits at render-time. Only edited bubbles get rows.
CREATE TABLE IF NOT EXISTS translation_edits (
    translation_id  BIGINT NOT NULL REFERENCES translations(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    edited_text     TEXT NOT NULL,
    edited_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (translation_id, page_index, bubble_idx)
);

-- ── Library (per-user grouping of materials) ────────────────────────
-- A library_entry is the user's "this is the manga I'm tracking" —
-- bookmark, last-read position, "Continue reading" hang here. Each
-- entry links one or more materials (cross-source).

CREATE TABLE IF NOT EXISTS library_entries (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title               TEXT NOT NULL,
    cover_url           TEXT,
    primary_material_id BIGINT REFERENCES materials(id) ON DELETE SET NULL,

    -- Reading language preference. Drives the hub's chapter list:
    -- chapters whose only available version is `target_lang` show
    -- "Read"; others show "Translate" + spawn-on-click. NULL means
    -- "user has not chosen yet" — the hub asks at first open.
    target_lang         TEXT,
    -- When TRUE, the watcher auto-spawns a translation as soon as a
    -- new chapter lands and `target_lang` differs from the source's
    -- native langs. Defaults to FALSE so casual users don't burn
    -- LLM quota on every new chapter.
    auto_translate      BOOLEAN NOT NULL DEFAULT FALSE,

    -- Reading status — the verb the user applies to the manga.
    --   reading   actively reading; default after first "Add".
    --   plan      saved to read later.
    --   on_hold   paused mid-series.
    --   done      finished reading the available run.
    --   dropped   no longer interested; hidden from default views.
    -- Replaces the legacy boolean `bookmarked` flag entirely; the
    -- library UI filters by status enum.
    status              TEXT NOT NULL DEFAULT 'reading'
                          CHECK (status IN ('reading','plan','on_hold','done','dropped')),

    last_read_at        TIMESTAMPTZ,
    last_chapter_ref    JSONB,    -- {material_id, chapter_id, label, position}
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_library_user ON library_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_library_user_status
    ON library_entries(user_id, status);

CREATE TABLE IF NOT EXISTS library_materials (
    entry_id     BIGINT NOT NULL REFERENCES library_entries(id) ON DELETE CASCADE,
    material_id  BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    -- Denormalized owner for the per-user uniqueness constraint below.
    -- Kept in sync with library_entries.user_id at insert time.
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    link_origin  TEXT NOT NULL CHECK (link_origin IN ('primary','auto','manual')),
    linked_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entry_id, material_id)
);
-- A material appears in at most one library_entry per user. Different
-- users may each have their own entry referencing the same material.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_library_material_per_user
    ON library_materials (user_id, material_id);

-- ── Glossary ────────────────────────────────────────────────────────
-- Per-user glossary: replaces the old per-project glossary. User
-- entries cover their own customizations; community_glossary is the
-- default body. The two are merged at glossary_fp computation time.

CREATE TABLE IF NOT EXISTS user_glossary (
    id           BIGSERIAL PRIMARY KEY,
    owner_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    notes        TEXT,
    source_term_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', source_term)) STORED,
    UNIQUE (owner_id, source_lang, target_lang, source_term)
);
CREATE INDEX IF NOT EXISTS idx_user_glossary_text
    ON user_glossary USING GIN (source_term_tsv);

-- Community glossary scaffold. Phase 1: schema-only, admin-seeded.
-- Phase 2: voting UI exposed to users. `material_id` NULL = global
-- term; non-null = scoped to one material (e.g. character name).
CREATE TABLE IF NOT EXISTS community_glossary (
    id           BIGSERIAL PRIMARY KEY,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    material_id  BIGINT REFERENCES materials(id) ON DELETE CASCADE,
    vote_score   INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_lang, target_lang, source_term, material_id)
);
CREATE INDEX IF NOT EXISTS idx_community_glossary_lookup
    ON community_glossary(source_lang, target_lang, material_id);

-- ── Translator memory ───────────────────────────────────────────────
-- Per-user, per-material, per-target-lang knowledge cards + accumulating
-- chapter briefs. Replaces the project-cũ idea of "project settings +
-- glossary" without coupling to a project entity.
--
-- Mental model:
--   characters / world / style / glossary cards are the long-lived
--     facts the user (and the agent) build up across chapters.
--   style_refs marks translations (or raw chapters at target_lang) the
--     user wants the agent to imitate. Phase 1: schema-only field;
--     UI + LLM context wiring lands later.
--   translator_memory_briefs holds per-chapter ChapterBrief output
--     so a sliding window of recent chapters can be injected when
--     translating chapter N+1.
--
-- JSONB shapes (Phase 1 — flexible until UX settles):
--   characters  [{name, aliases[], pronouns:{self,other}, role, notes, locked?}]
--   world       {setting, factions, places, terminology, notes}
--   style       {tone, formality, address_style, sfx, onomatopoeia, custom?}
--   glossary    [{source_term, target_term, notes, locked?}]
--   style_refs  [{kind: 'translation'|'chapter', id, label, weight}]
--
-- Suggestions queue (agent → user) lives inside `characters` /
-- `glossary` rows themselves via a `pending` flag, not a separate
-- table — keeps reads single-row.

CREATE TABLE IF NOT EXISTS translator_memory (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    material_id   BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    source_lang   TEXT NOT NULL,
    target_lang   TEXT NOT NULL,

    characters    JSONB NOT NULL DEFAULT '[]'::jsonb,
    world         JSONB NOT NULL DEFAULT '{}'::jsonb,
    style         JSONB NOT NULL DEFAULT '{}'::jsonb,
    glossary      JSONB NOT NULL DEFAULT '[]'::jsonb,
    style_refs    JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Highest chapter.position the agent has folded into this memory.
    -- Used to gate "learn from chapter N" reruns and to decide which
    -- briefs are still in the sliding window.
    last_chapter_id  BIGINT REFERENCES chapters(id) ON DELETE SET NULL,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, material_id, target_lang)
);
CREATE INDEX IF NOT EXISTS idx_translator_memory_user
    ON translator_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_translator_memory_material
    ON translator_memory(material_id);

-- Sliding-window chapter briefs. One row per (memory, chapter).
-- `brief_json` holds the ChapterBrief shape (glossary, style_notes,
-- key_notes, characters, noise_keys, noise_pages) emitted by the
-- per-chapter storyboard vision pass. `summary` is denormalised from
-- the first style_note line for list views + FTS.
CREATE TABLE IF NOT EXISTS translator_memory_briefs (
    memory_id    BIGINT NOT NULL REFERENCES translator_memory(id) ON DELETE CASCADE,
    chapter_id   BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    brief_json   JSONB NOT NULL,
    summary      TEXT,
    summary_tsv  tsvector
        GENERATED ALWAYS AS (to_tsvector('simple', coalesce(summary, ''))) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (memory_id, chapter_id)
);
CREATE INDEX IF NOT EXISTS idx_tm_briefs_chapter
    ON translator_memory_briefs(chapter_id);
CREATE INDEX IF NOT EXISTS idx_tm_briefs_summary_tsv
    ON translator_memory_briefs USING GIN (summary_tsv);

-- ── Worker coordination ─────────────────────────────────────────────
-- One queue table for the pipeline. `target_kind` is the discriminator:
--   'chapter'     → prepare, scan (chapter-level work)
--   'draft'       → translate, render-default (lang/glossary-level)
--   'translation' → render-edits (per-user when edits exist)
-- Stage transitions live in worker code; the queue itself is dumb.

CREATE TABLE IF NOT EXISTS tasks (
    target_kind  TEXT NOT NULL CHECK (target_kind IN ('chapter','draft','translation')),
    target_id    BIGINT NOT NULL,
    stage        TEXT NOT NULL CHECK (stage IN ('prepare','scan','translate','render')),
    claimed_by   TEXT,
    claimed_at   TIMESTAMPTZ,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    PRIMARY KEY (target_kind, target_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(stage, claimed_by, claimed_at);

-- Wake LISTEN'ing workers when a task becomes claimable. Payload
-- format: '<target_kind>:<target_id>' so each worker can filter to
-- its own (kind, stage).
CREATE OR REPLACE FUNCTION notify_task_ready() RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        PERFORM pg_notify('typoon_task_' || NEW.stage,
                          NEW.target_kind || ':' || NEW.target_id::text);
    ELSIF TG_OP = 'UPDATE'
          AND OLD.claimed_by IS NOT NULL
          AND NEW.claimed_by IS NULL THEN
        PERFORM pg_notify('typoon_task_' || NEW.stage,
                          NEW.target_kind || ':' || NEW.target_id::text);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tasks_notify_ready ON tasks;
CREATE TRIGGER tasks_notify_ready
    AFTER INSERT OR UPDATE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION notify_task_ready();

-- ── Quota usage log ─────────────────────────────────────────────────
-- One row per LLM-costing event. Cache HITS do not insert; quota
-- counts only what the user actually paid for. Time-windowed counters
-- power the quota chip in the UI.

CREATE TABLE IF NOT EXISTS chapter_consumes (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    translation_id  BIGINT REFERENCES translations(id) ON DELETE SET NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('draft_create','render_create')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chapter_consumes_user_time
    ON chapter_consumes(user_id, created_at DESC);

-- ── Reports ─────────────────────────────────────────────────────────
-- User-submitted reports flow into `reports` (intake). Admin
-- moderation is logged separately in `moderation_actions` (audit).
-- The two are joined via `report_id`; one report can trigger many
-- actions (takedown then restore then re-takedown) and one action
-- can happen without a prior report (admin-initiated cleanup).
--
-- The visibility flip (target row's takedown_at / takedown_reason)
-- still lives on `translation_drafts` / `translations` — those
-- columns drive the actual read-path filter. `moderation_actions`
-- is the audit log of how the flag got there.

CREATE TABLE IF NOT EXISTS reports (
    id              BIGSERIAL PRIMARY KEY,
    reporter_id     BIGINT REFERENCES users(id) ON DELETE SET NULL,
    -- Free-form label for the reporter — used when the reporter is
    -- an anonymous DMCA agent (email/handle) instead of an app user.
    -- For authenticated reports we still populate this with the
    -- display_name snapshot at submit time so the admin queue stays
    -- readable after account deletion.
    reporter_label  TEXT NOT NULL,

    target_kind     TEXT NOT NULL
        CHECK (target_kind IN ('material','chapter','draft','translation')),
    target_id       BIGINT NOT NULL,
    scope_guild_id  TEXT,

    kind            TEXT NOT NULL DEFAULT 'dmca'
        CHECK (kind IN ('dmca','abuse','quality','other')),
    reason          TEXT NOT NULL,

    status          TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open','reviewing','resolved','dismissed')),

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    resolved_by     BIGINT REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_status
    ON reports(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_target
    ON reports(target_kind, target_id);

CREATE TABLE IF NOT EXISTS moderation_actions (
    id              BIGSERIAL PRIMARY KEY,
    -- NULL when admin acted without a triggering report (proactive
    -- cleanup) or when the report was hard-deleted.
    report_id       BIGINT REFERENCES reports(id) ON DELETE SET NULL,

    target_kind     TEXT NOT NULL
        CHECK (target_kind IN ('material','chapter','draft','translation')),
    target_id       BIGINT NOT NULL,

    action          TEXT NOT NULL
        CHECK (action IN ('takedown','restore','delete')),
    reason          TEXT NOT NULL,

    actor_id        BIGINT REFERENCES users(id) ON DELETE SET NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_moderation_actions_target
    ON moderation_actions(target_kind, target_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_moderation_actions_report
    ON moderation_actions(report_id) WHERE report_id IS NOT NULL;

-- ── updated_at maintenance ──────────────────────────────────────────
-- Postgres has no "ON UPDATE" trigger sugar — keep them explicit.

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS materials_touch_updated_at ON materials;
CREATE TRIGGER materials_touch_updated_at
    BEFORE UPDATE ON materials
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS chapters_touch_updated_at ON chapters;
CREATE TRIGGER chapters_touch_updated_at
    BEFORE UPDATE ON chapters
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS drafts_touch_updated_at ON translation_drafts;
CREATE TRIGGER drafts_touch_updated_at
    BEFORE UPDATE ON translation_drafts
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS translations_touch_updated_at ON translations;
CREATE TRIGGER translations_touch_updated_at
    BEFORE UPDATE ON translations
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS library_entries_touch_updated_at ON library_entries;
CREATE TRIGGER library_entries_touch_updated_at
    BEFORE UPDATE ON library_entries
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS translator_memory_touch_updated_at ON translator_memory;
CREATE TRIGGER translator_memory_touch_updated_at
    BEFORE UPDATE ON translator_memory
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

DROP TRIGGER IF EXISTS tm_briefs_touch_updated_at ON translator_memory_briefs;
CREATE TRIGGER tm_briefs_touch_updated_at
    BEFORE UPDATE ON translator_memory_briefs
    FOR EACH ROW
    WHEN (OLD.updated_at IS NOT DISTINCT FROM NEW.updated_at)
    EXECUTE FUNCTION touch_updated_at();

-- Cascade chapter writes to material.updated_at — drives "Cập nhật N
-- ngày trước" badge on library cards.
CREATE OR REPLACE FUNCTION touch_material_via_chapter() RETURNS TRIGGER AS $$
BEGIN
    UPDATE materials SET updated_at = NOW()
        WHERE id = COALESCE(NEW.material_id, OLD.material_id);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chapters_touch_material ON chapters;
CREATE TRIGGER chapters_touch_material
    AFTER INSERT OR UPDATE OR DELETE ON chapters
    FOR EACH ROW
    EXECUTE FUNCTION touch_material_via_chapter();

-- Task lifecycle bumps the draft / translation it targets — UI uses
-- updated_at for freshness on the progress chip.
CREATE OR REPLACE FUNCTION touch_target_via_task() RETURNS TRIGGER AS $$
DECLARE
    tk TEXT := COALESCE(NEW.target_kind, OLD.target_kind);
    ti BIGINT := COALESCE(NEW.target_id, OLD.target_id);
BEGIN
    IF tk = 'chapter' THEN
        UPDATE chapters SET updated_at = NOW() WHERE id = ti;
    ELSIF tk = 'draft' THEN
        UPDATE translation_drafts SET updated_at = NOW() WHERE id = ti;
    ELSIF tk = 'translation' THEN
        UPDATE translations SET updated_at = NOW() WHERE id = ti;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tasks_touch_target ON tasks;
CREATE TRIGGER tasks_touch_target
    AFTER INSERT OR UPDATE OR DELETE ON tasks
    FOR EACH ROW
    EXECUTE FUNCTION touch_target_via_task();
