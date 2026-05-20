-- D1 / SQLite schema for Typoon — CF Native
--
-- Converted from Postgres schema v26.
-- Source of truth for `wrangler d1 execute typoon-db --file=schema.sql`
--
-- Removed vs Postgres:
--   tasks, stage_pause          → CF Workflows (never needed in DB)
--   material_inbox              → presigned R2 multipart, no inbox
--   pg_trgm + GIST indexes      → Levenshtein in Worker JS
--   tsvector GENERATED columns  → FTS5 virtual tables
--   pg_notify triggers          → CF Durable Objects WebSocket
--   Materialized view           → regular table refreshed by cron Worker
--   JSONB GIN indexes           → json_extract() queries
--
-- Conventions:
--   INTEGER PRIMARY KEY         → SQLite rowid alias, auto-increment
--   TEXT DEFAULT (datetime())   → ISO-8601 UTC string
--   TEXT (for arrays/objects)   → JSON serialized, use json_extract()
--   BOOLEAN                     → INTEGER (0/1)

-- PRAGMA journal_mode = WAL;
-- PRAGMA foreign_keys = ON;

-- ── Schema version ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '1');

-- ── Identity ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
  id                    INTEGER PRIMARY KEY,
  display_name          TEXT    NOT NULL,
  avatar_url            TEXT,
  email                 TEXT,
  preferred_target_lang TEXT,
  created_at            TEXT    NOT NULL DEFAULT (datetime('now')),
  last_login_at         TEXT
);

CREATE TABLE IF NOT EXISTS identities (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider    TEXT    NOT NULL,
  external_id TEXT    NOT NULL,
  metadata    TEXT,                        -- JSON
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (provider, external_id)
);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities(user_id);

CREATE TABLE IF NOT EXISTS api_tokens (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name        TEXT    NOT NULL,
  token_hash  TEXT    NOT NULL UNIQUE,
  prefix      TEXT    NOT NULL,
  scopes      TEXT    NOT NULL DEFAULT '[]',   -- JSON array: ["worker","admin"]
  last_used   TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  revoked_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_tokens_user_active
  ON api_tokens(user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_api_tokens_prefix_active
  ON api_tokens(prefix)  WHERE revoked_at IS NULL;

-- ── Works ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS works (
  id         INTEGER PRIMARY KEY,
  cross_refs TEXT,                          -- JSON: {"mdex_uuid":"…","anilist":12345}
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS work_redirects (
  old_id    INTEGER PRIMARY KEY,
  new_id    INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
  merged_at TEXT    NOT NULL DEFAULT (datetime('now')),
  CHECK (old_id != new_id)
);
CREATE INDEX IF NOT EXISTS idx_work_redirects_new ON work_redirects(new_id);

-- Collapse transitive redirects: if A→B exists and we insert B→C,
-- rewrite A→C so lookups never chain.
CREATE TRIGGER IF NOT EXISTS collapse_work_redirects
AFTER INSERT ON work_redirects
BEGIN
  UPDATE work_redirects SET new_id = NEW.new_id WHERE new_id = NEW.old_id;
END;

CREATE TRIGGER IF NOT EXISTS works_updated_at
AFTER UPDATE ON works FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE works SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS work_chapters (
  id          INTEGER PRIMARY KEY,
  work_id     INTEGER NOT NULL REFERENCES works(id) ON DELETE CASCADE,
  number_norm TEXT    NOT NULL,
  label       TEXT,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (work_id, number_norm)
);
CREATE INDEX IF NOT EXISTS idx_work_chapters_work ON work_chapters(work_id);

-- ── Materials ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS materials (
  id            INTEGER PRIMARY KEY,
  imported_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
  origin        TEXT    NOT NULL CHECK (origin IN ('source','extension','upload')),
  work_id       INTEGER NOT NULL REFERENCES works(id),
  source        TEXT,
  upstream_ref  TEXT,
  title         TEXT    NOT NULL,
  cover_url     TEXT,
  description   TEXT,
  author        TEXT,
  status        TEXT,
  languages     TEXT    NOT NULL DEFAULT '[]',   -- JSON string[]
  title_native  TEXT,
  title_alt     TEXT,                             -- JSON string[]
  cross_refs    TEXT,                             -- JSON object
  title_locale  TEXT,                             -- JSON object: {"vi":"…","en":"…"}
  start_year    INTEGER,
  nsfw          INTEGER NOT NULL DEFAULT 0,       -- 0=false 1=true
  created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_materials_source_ref
  ON materials(source, upstream_ref) WHERE source IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uniq_upload_material_per_user_work
  ON materials(imported_by, work_id) WHERE origin = 'upload';
CREATE INDEX IF NOT EXISTS idx_materials_imported_by ON materials(imported_by);
CREATE INDEX IF NOT EXISTS idx_materials_work        ON materials(work_id);
CREATE INDEX IF NOT EXISTS idx_materials_title_native
  ON materials(title_native) WHERE title_native IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS materials_updated_at
AFTER UPDATE ON materials FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE materials SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- ── Material link votes ─────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS material_link_votes (
  material_a_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
  material_b_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
  voter_id      INTEGER NOT NULL REFERENCES users(id)     ON DELETE CASCADE,
  vote          INTEGER NOT NULL CHECK (vote IN (-1, 1)),
  voted_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (material_a_id, material_b_id, voter_id),
  CHECK (material_a_id < material_b_id)
);
CREATE INDEX IF NOT EXISTS idx_link_votes_a ON material_link_votes(material_a_id);
CREATE INDEX IF NOT EXISTS idx_link_votes_b ON material_link_votes(material_b_id);

-- Regular table replacing the Postgres materialized view.
-- Refreshed by a scheduled cron Worker (every 5 minutes).
CREATE TABLE IF NOT EXISTS material_links (
  material_a_id INTEGER NOT NULL,
  material_b_id INTEGER NOT NULL,
  score         INTEGER NOT NULL DEFAULT 0,
  total_votes   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (material_a_id, material_b_id)
);
CREATE INDEX IF NOT EXISTS idx_links_a ON material_links(material_a_id);
CREATE INDEX IF NOT EXISTS idx_links_b ON material_links(material_b_id);

CREATE TABLE IF NOT EXISTS material_split_votes (
  material_id INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
  voter_id    INTEGER NOT NULL REFERENCES users(id)     ON DELETE CASCADE,
  vote        INTEGER NOT NULL CHECK (vote IN (-1, 1)),
  voted_at    TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (material_id, voter_id)
);
CREATE INDEX IF NOT EXISTS idx_split_votes_material ON material_split_votes(material_id);

CREATE TABLE IF NOT EXISTS material_link_actions (
  id             INTEGER PRIMARY KEY,
  actor_id       INTEGER NOT NULL REFERENCES users(id)       ON DELETE CASCADE,
  kind           TEXT    NOT NULL CHECK (kind IN ('force_link','force_unlink')),
  material_a_id  INTEGER NOT NULL REFERENCES materials(id)   ON DELETE CASCADE,
  material_b_id  INTEGER          REFERENCES materials(id)   ON DELETE CASCADE,
  target_work_id INTEGER NOT NULL REFERENCES works(id)       ON DELETE CASCADE,
  reversed_at    TEXT,
  created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_link_actions_actor_recent
  ON material_link_actions(actor_id, created_at DESC) WHERE reversed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_link_actions_material_a
  ON material_link_actions(material_a_id);

-- ── Chapters ────────────────────────────────────────────────────────
--
-- R2 key conventions (set by pipeline stages):
--   prepared_prefix  "prepared/{chapter_id}/"  — set after prepare
--   masks_prefix     "mask/{chapter_id}/"       — set after scan

CREATE TABLE IF NOT EXISTS chapters (
  id               INTEGER PRIMARY KEY,
  material_id      INTEGER NOT NULL REFERENCES materials(id)      ON DELETE CASCADE,
  work_chapter_id  INTEGER NOT NULL REFERENCES work_chapters(id),
  position         INTEGER NOT NULL,
  label            TEXT,
  upstream_url     TEXT,
  source_lang      TEXT,
  prepared_prefix  TEXT,   -- R2 prefix; NULL until prepare stage completes
  masks_prefix     TEXT,   -- R2 prefix; NULL until scan stage completes
  page_count       INTEGER NOT NULL DEFAULT 0,
  created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (material_id, position),
  UNIQUE (material_id, upstream_url)
);
CREATE INDEX IF NOT EXISTS idx_chapters_material
  ON chapters(material_id, position);
CREATE INDEX IF NOT EXISTS idx_chapters_work_chapter
  ON chapters(work_chapter_id);

CREATE TRIGGER IF NOT EXISTS chapters_updated_at
AFTER UPDATE ON chapters FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE chapters SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- Propagate chapter changes to material.updated_at
CREATE TRIGGER IF NOT EXISTS chapters_touch_material_insert
AFTER INSERT ON chapters
BEGIN
  UPDATE materials SET updated_at = datetime('now') WHERE id = NEW.material_id;
END;

CREATE TRIGGER IF NOT EXISTS chapters_touch_material_update
AFTER UPDATE ON chapters
BEGIN
  UPDATE materials SET updated_at = datetime('now') WHERE id = NEW.material_id;
END;

CREATE TRIGGER IF NOT EXISTS chapters_touch_material_delete
AFTER DELETE ON chapters
BEGIN
  UPDATE materials SET updated_at = datetime('now') WHERE id = OLD.material_id;
END;

-- ── Scan output (chapter-scoped, shared across all translations) ────

CREATE TABLE IF NOT EXISTS bubbles (
  chapter_id  INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
  page_index  INTEGER NOT NULL,
  bubble_idx  INTEGER NOT NULL,
  source_text TEXT    NOT NULL,
  confidence  REAL    NOT NULL,
  shape_kind  TEXT    NOT NULL DEFAULT 'dialogue'
    CHECK (shape_kind IN ('dialogue','burst')),
  PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

CREATE VIRTUAL TABLE IF NOT EXISTS bubbles_fts USING fts5(
  source_text,
  chapter_id UNINDEXED,
  page_index UNINDEXED,
  bubble_idx UNINDEXED,
  content    = 'bubbles',
  content_rowid = 'rowid'
);

CREATE TRIGGER IF NOT EXISTS bubbles_fts_insert AFTER INSERT ON bubbles BEGIN
  INSERT INTO bubbles_fts(rowid, source_text, chapter_id, page_index, bubble_idx)
  VALUES (new.rowid, new.source_text, new.chapter_id, new.page_index, new.bubble_idx);
END;
CREATE TRIGGER IF NOT EXISTS bubbles_fts_update AFTER UPDATE ON bubbles BEGIN
  UPDATE bubbles_fts SET source_text = new.source_text WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS bubbles_fts_delete AFTER DELETE ON bubbles BEGIN
  DELETE FROM bubbles_fts WHERE rowid = old.rowid;
END;

CREATE TABLE IF NOT EXISTS bubble_geometry (
  chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
  page_index INTEGER NOT NULL,
  bubble_idx INTEGER NOT NULL,
  polygon    TEXT    NOT NULL,   -- JSON [[x,y],…]
  PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

CREATE TABLE IF NOT EXISTS page_geometry (
  chapter_id INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
  page_index INTEGER NOT NULL,
  width      INTEGER NOT NULL,
  height     INTEGER NOT NULL,
  PRIMARY KEY (chapter_id, page_index)
);

-- ── Translation drafts (lang + glossary level) ──────────────────────

CREATE TABLE IF NOT EXISTS translation_drafts (
  id              INTEGER PRIMARY KEY,
  chapter_id      INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
  source_lang     TEXT    NOT NULL,
  target_lang     TEXT    NOT NULL,
  glossary_fp     TEXT    NOT NULL,
  llm_model       TEXT    NOT NULL,
  created_by      INTEGER REFERENCES users(id) ON DELETE SET NULL,
  takedown_at     TEXT,
  takedown_reason TEXT,
  state           TEXT    NOT NULL DEFAULT 'pending'
    CHECK (state IN ('pending','running','done','error','blocked')),
  error_message   TEXT,
  -- Progress events emitted via DO WebSocket; stored here for late-joiners.
  progress_stage  TEXT,
  progress_index  INTEGER,
  progress_total  INTEGER,
  -- R2 key for the rendered BNL archive (set by pipeline finalize step)
  archive_key     TEXT,
  rendered_at     TEXT,
  created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_drafts_cache
  ON translation_drafts(chapter_id, source_lang, target_lang, glossary_fp)
  WHERE takedown_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_drafts_creator ON translation_drafts(created_by);
CREATE INDEX IF NOT EXISTS idx_drafts_chapter  ON translation_drafts(chapter_id);

CREATE TRIGGER IF NOT EXISTS drafts_updated_at
AFTER UPDATE ON translation_drafts FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE translation_drafts SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS translation_draft_bubbles (
  draft_id        INTEGER NOT NULL REFERENCES translation_drafts(id) ON DELETE CASCADE,
  page_index      INTEGER NOT NULL,
  bubble_idx      INTEGER NOT NULL,
  translated_text TEXT    NOT NULL,
  kind            TEXT    NOT NULL CHECK (kind IN ('dialogue','sfx','skip')),
  PRIMARY KEY (draft_id, page_index, bubble_idx)
);

CREATE VIRTUAL TABLE IF NOT EXISTS translation_draft_bubbles_fts USING fts5(
  translated_text,
  draft_id UNINDEXED,
  content       = 'translation_draft_bubbles',
  content_rowid = 'rowid'
);

CREATE TRIGGER IF NOT EXISTS tdb_fts_insert AFTER INSERT ON translation_draft_bubbles BEGIN
  INSERT INTO translation_draft_bubbles_fts(rowid, translated_text, draft_id)
  VALUES (new.rowid, new.translated_text, new.draft_id);
END;
CREATE TRIGGER IF NOT EXISTS tdb_fts_update AFTER UPDATE ON translation_draft_bubbles BEGIN
  UPDATE translation_draft_bubbles_fts SET translated_text = new.translated_text
  WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS tdb_fts_delete AFTER DELETE ON translation_draft_bubbles BEGIN
  DELETE FROM translation_draft_bubbles_fts WHERE rowid = old.rowid;
END;

CREATE TABLE IF NOT EXISTS draft_briefs (
  draft_id   INTEGER PRIMARY KEY REFERENCES translation_drafts(id) ON DELETE CASCADE,
  brief_json TEXT NOT NULL,   -- JSON (full BriefIndex shape)
  summary    TEXT,
  terms_text TEXT,
  facts_text TEXT,
  rules_text TEXT,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS draft_briefs_fts USING fts5(
  summary, terms_text, facts_text, rules_text,
  content       = 'draft_briefs',
  content_rowid = 'rowid'
);

CREATE TRIGGER IF NOT EXISTS draft_briefs_fts_insert AFTER INSERT ON draft_briefs BEGIN
  INSERT INTO draft_briefs_fts(rowid, summary, terms_text, facts_text, rules_text)
  VALUES (new.rowid, new.summary, new.terms_text, new.facts_text, new.rules_text);
END;
CREATE TRIGGER IF NOT EXISTS draft_briefs_fts_update AFTER UPDATE ON draft_briefs BEGIN
  UPDATE draft_briefs_fts
  SET summary = new.summary, terms_text = new.terms_text,
      facts_text = new.facts_text, rules_text = new.rules_text
  WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS draft_briefs_fts_delete AFTER DELETE ON draft_briefs BEGIN
  DELETE FROM draft_briefs_fts WHERE rowid = old.rowid;
END;

-- ── Translations (per-user wrapper) ────────────────────────────────

CREATE TABLE IF NOT EXISTS translations (
  id               INTEGER PRIMARY KEY,
  work_chapter_id  INTEGER NOT NULL REFERENCES work_chapters(id)     ON DELETE CASCADE,
  owner_id         INTEGER NOT NULL REFERENCES users(id)             ON DELETE CASCADE,
  target_lang      TEXT    NOT NULL,
  draft_id         INTEGER NOT NULL REFERENCES translation_drafts(id) ON DELETE CASCADE,
  shared           INTEGER NOT NULL DEFAULT 1,   -- 0=private 1=shared
  -- NULL → fall back to draft.archive_key (user has no custom render)
  archive_key      TEXT,
  rendered_at      TEXT,
  takedown_at      TEXT,
  takedown_reason  TEXT,
  created_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at       TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (work_chapter_id, owner_id, draft_id)
);
CREATE INDEX IF NOT EXISTS idx_translations_owner       ON translations(owner_id);
CREATE INDEX IF NOT EXISTS idx_translations_work_chapter ON translations(work_chapter_id);
CREATE INDEX IF NOT EXISTS idx_translations_draft        ON translations(draft_id);
CREATE INDEX IF NOT EXISTS idx_translations_recent
  ON translations(created_at DESC) WHERE takedown_at IS NULL;

CREATE TRIGGER IF NOT EXISTS translations_updated_at
AFTER UPDATE ON translations FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE translations SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- Sparse edits over the shared draft (only edited bubbles get rows)
CREATE TABLE IF NOT EXISTS translation_edits (
  translation_id INTEGER NOT NULL REFERENCES translations(id) ON DELETE CASCADE,
  page_index     INTEGER NOT NULL,
  bubble_idx     INTEGER NOT NULL,
  edited_text    TEXT    NOT NULL,
  edited_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (translation_id, page_index, bubble_idx)
);

-- ── Library ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS library_entries (
  id          INTEGER PRIMARY KEY,
  user_id     INTEGER NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
  work_id     INTEGER NOT NULL REFERENCES works(id)  ON DELETE CASCADE,
  target_lang TEXT    NOT NULL DEFAULT 'vi',
  status      TEXT    NOT NULL DEFAULT 'reading'
    CHECK (status IN ('reading','plan','done','dropped')),
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_library_user        ON library_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_library_user_status ON library_entries(user_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_library_entries_user_work
  ON library_entries(user_id, work_id);

CREATE TRIGGER IF NOT EXISTS library_entries_updated_at
AFTER UPDATE ON library_entries FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE library_entries SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS library_materials (
  entry_id    INTEGER NOT NULL REFERENCES library_entries(id) ON DELETE CASCADE,
  material_id INTEGER NOT NULL REFERENCES materials(id)       ON DELETE CASCADE,
  user_id     INTEGER NOT NULL REFERENCES users(id)           ON DELETE CASCADE,
  link_origin TEXT    NOT NULL CHECK (link_origin IN ('auto','manual')),
  linked_at   TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (entry_id, material_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_library_material_per_user
  ON library_materials(user_id, material_id);

CREATE TABLE IF NOT EXISTS reading_history (
  user_id          INTEGER NOT NULL REFERENCES users(id)          ON DELETE CASCADE,
  work_chapter_id  INTEGER NOT NULL REFERENCES work_chapters(id)  ON DELETE CASCADE,
  last_material_id INTEGER          REFERENCES materials(id)      ON DELETE SET NULL,
  translation_id   INTEGER          REFERENCES translations(id)   ON DELETE SET NULL,
  last_read_at     TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (user_id, work_chapter_id)
);
CREATE INDEX IF NOT EXISTS idx_reading_history_user_time
  ON reading_history(user_id, last_read_at DESC);

-- ── Glossary ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_glossary (
  id          INTEGER PRIMARY KEY,
  owner_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  source_lang TEXT    NOT NULL,
  target_lang TEXT    NOT NULL,
  source_term TEXT    NOT NULL,
  target_term TEXT    NOT NULL,
  notes       TEXT,
  UNIQUE (owner_id, source_lang, target_lang, source_term)
);

CREATE VIRTUAL TABLE IF NOT EXISTS user_glossary_fts USING fts5(
  source_term,
  content       = 'user_glossary',
  content_rowid = 'rowid'
);

CREATE TRIGGER IF NOT EXISTS user_glossary_fts_insert AFTER INSERT ON user_glossary BEGIN
  INSERT INTO user_glossary_fts(rowid, source_term) VALUES (new.rowid, new.source_term);
END;
CREATE TRIGGER IF NOT EXISTS user_glossary_fts_update AFTER UPDATE ON user_glossary BEGIN
  UPDATE user_glossary_fts SET source_term = new.source_term WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS user_glossary_fts_delete AFTER DELETE ON user_glossary BEGIN
  DELETE FROM user_glossary_fts WHERE rowid = old.rowid;
END;

CREATE TABLE IF NOT EXISTS community_glossary (
  id          INTEGER PRIMARY KEY,
  source_lang TEXT    NOT NULL,
  target_lang TEXT    NOT NULL,
  source_term TEXT    NOT NULL,
  target_term TEXT    NOT NULL,
  material_id INTEGER REFERENCES materials(id) ON DELETE CASCADE,
  vote_score  INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (source_lang, target_lang, source_term, material_id)
);
CREATE INDEX IF NOT EXISTS idx_community_glossary_lookup
  ON community_glossary(source_lang, target_lang, material_id);

-- ── Translator memory ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS translator_memory (
  id              INTEGER PRIMARY KEY,
  user_id         INTEGER NOT NULL REFERENCES users(id)      ON DELETE CASCADE,
  material_id     INTEGER NOT NULL REFERENCES materials(id)  ON DELETE CASCADE,
  source_lang     TEXT    NOT NULL,
  target_lang     TEXT    NOT NULL,
  characters      TEXT    NOT NULL DEFAULT '[]',   -- JSON
  world           TEXT    NOT NULL DEFAULT '{}',   -- JSON
  style           TEXT    NOT NULL DEFAULT '{}',   -- JSON
  glossary        TEXT    NOT NULL DEFAULT '[]',   -- JSON
  style_refs      TEXT    NOT NULL DEFAULT '[]',   -- JSON
  last_chapter_id INTEGER REFERENCES chapters(id) ON DELETE SET NULL,
  created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
  UNIQUE (user_id, material_id, target_lang)
);
CREATE INDEX IF NOT EXISTS idx_translator_memory_user     ON translator_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_translator_memory_material ON translator_memory(material_id);

CREATE TRIGGER IF NOT EXISTS translator_memory_updated_at
AFTER UPDATE ON translator_memory FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE translator_memory SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS translator_memory_briefs (
  memory_id  INTEGER NOT NULL REFERENCES translator_memory(id) ON DELETE CASCADE,
  chapter_id INTEGER NOT NULL REFERENCES chapters(id)          ON DELETE CASCADE,
  brief_json TEXT    NOT NULL,   -- JSON BriefIndex shape
  summary    TEXT,
  created_at TEXT    NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (memory_id, chapter_id)
);
CREATE INDEX IF NOT EXISTS idx_tm_briefs_chapter ON translator_memory_briefs(chapter_id);

CREATE VIRTUAL TABLE IF NOT EXISTS translator_memory_briefs_fts USING fts5(
  summary,
  content       = 'translator_memory_briefs',
  content_rowid = 'rowid'
);

CREATE TRIGGER IF NOT EXISTS tm_briefs_fts_insert AFTER INSERT ON translator_memory_briefs BEGIN
  INSERT INTO translator_memory_briefs_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS tm_briefs_fts_update AFTER UPDATE ON translator_memory_briefs BEGIN
  UPDATE translator_memory_briefs_fts SET summary = new.summary WHERE rowid = old.rowid;
END;
CREATE TRIGGER IF NOT EXISTS tm_briefs_fts_delete AFTER DELETE ON translator_memory_briefs BEGIN
  DELETE FROM translator_memory_briefs_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER IF NOT EXISTS tm_briefs_updated_at
AFTER UPDATE ON translator_memory_briefs FOR EACH ROW
WHEN NEW.updated_at = OLD.updated_at
BEGIN
  UPDATE translator_memory_briefs SET updated_at = datetime('now') WHERE id = NEW.id;
END;

-- ── Quota / Moderation ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS chapter_consumes (
  id             INTEGER PRIMARY KEY,
  user_id        INTEGER NOT NULL REFERENCES users(id)        ON DELETE CASCADE,
  translation_id INTEGER          REFERENCES translations(id) ON DELETE SET NULL,
  kind           TEXT    NOT NULL CHECK (kind IN ('draft_create','render_create')),
  created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chapter_consumes_user_time
  ON chapter_consumes(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS reports (
  id             INTEGER PRIMARY KEY,
  reporter_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
  reporter_label TEXT    NOT NULL,
  target_kind    TEXT    NOT NULL
    CHECK (target_kind IN ('material','chapter','draft','translation')),
  target_id      INTEGER NOT NULL,
  kind           TEXT    NOT NULL DEFAULT 'dmca'
    CHECK (kind IN ('dmca','abuse','quality','other')),
  reason         TEXT    NOT NULL,
  status         TEXT    NOT NULL DEFAULT 'open'
    CHECK (status IN ('open','reviewing','resolved','dismissed')),
  created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
  resolved_at    TEXT,
  resolved_by    INTEGER REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_target ON reports(target_kind, target_id);

CREATE TABLE IF NOT EXISTS moderation_actions (
  id          INTEGER PRIMARY KEY,
  report_id   INTEGER REFERENCES reports(id) ON DELETE SET NULL,
  target_kind TEXT NOT NULL
    CHECK (target_kind IN ('material','chapter','draft','translation')),
  target_id   INTEGER NOT NULL,
  action      TEXT    NOT NULL CHECK (action IN ('takedown','restore','delete')),
  reason      TEXT    NOT NULL,
  actor_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
  created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_moderation_actions_target
  ON moderation_actions(target_kind, target_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_moderation_actions_report
  ON moderation_actions(report_id) WHERE report_id IS NOT NULL;

-- Admin audit log — simplified action set for CF context.
-- Workflow-level ops handled by CF dashboard; only data mutations logged here.
CREATE TABLE IF NOT EXISTS admin_actions (
  id         INTEGER PRIMARY KEY,
  at         TEXT    NOT NULL DEFAULT (datetime('now')),
  actor_id   INTEGER REFERENCES users(id) ON DELETE SET NULL,
  action     TEXT    NOT NULL CHECK (action IN (
    'draft.takedown',       'draft.restore',
    'translation.takedown', 'translation.restore',
    'pipeline.cancel',      'pipeline.retry',
    'material.delete'
  )),
  target_ref TEXT    NOT NULL,   -- JSON: {kind, id, workflow_id?, idem_key?}
  reason     TEXT    NOT NULL CHECK (length(reason) >= 3),
  prev_state TEXT                -- JSON snapshot of affected row before mutation
);
CREATE INDEX IF NOT EXISTS idx_admin_actions_at
  ON admin_actions(at DESC);
CREATE INDEX IF NOT EXISTS idx_admin_actions_actor
  ON admin_actions(actor_id, at DESC) WHERE actor_id IS NOT NULL;
