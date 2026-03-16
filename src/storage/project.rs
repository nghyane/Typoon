use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

/// Self-contained project database.
///
/// One `project.db` per project holds all pipeline data:
/// - Pages (source image registration + detect status)
/// - Bubbles (detection + OCR output)
/// - Translations (translation + fit output)
/// - Translation history (audit trail)
/// - Glossary (per-project terms)
/// - Context notes (continuity notes)
///
/// Masks and rendered images are stored as files alongside the DB.
/// Their existence is checked at runtime — no tracking tables needed.
pub struct ProjectStore {
    conn: std::sync::Mutex<Connection>,
    root: PathBuf,
}

// ── Row types ──

#[derive(Debug, Clone)]
pub struct PageRow {
    pub chapter: usize,
    pub page: usize,
    pub image_hash: String,
    pub detected_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BubbleRow {
    pub chapter: usize,
    pub page: usize,
    pub idx: usize,
    pub source_text: String,
    pub polygon: String,
    pub drawable_area: String,
    pub det_confidence: f64,
    pub ocr_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct TranslationRow {
    pub chapter: usize,
    pub page: usize,
    pub idx: usize,
    pub translated_text: String,
    pub font_size_px: u32,
    pub line_height: f64,
    pub overflow: bool,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct GlossaryEntry {
    pub source_term: String,
    pub target_term: String,
    pub notes: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NoteRow {
    pub id: i64,
    pub chapter: usize,
    pub note_type: String,
    pub content: String,
}

// ── Schema ──

const SCHEMA: &str = "
    -- Source images
    CREATE TABLE IF NOT EXISTS pages (
        chapter     INTEGER NOT NULL,
        page        INTEGER NOT NULL,
        image_hash  TEXT NOT NULL,
        detected_at TEXT,
        PRIMARY KEY (chapter, page)
    );

    -- Detection + OCR output
    CREATE TABLE IF NOT EXISTS bubbles (
        chapter         INTEGER NOT NULL,
        page            INTEGER NOT NULL,
        idx             INTEGER NOT NULL,
        source_text     TEXT NOT NULL,
        polygon         TEXT NOT NULL,
        drawable_area   TEXT NOT NULL,
        det_confidence  REAL NOT NULL,
        ocr_confidence  REAL NOT NULL,
        PRIMARY KEY (chapter, page, idx),
        FOREIGN KEY (chapter, page) REFERENCES pages(chapter, page) ON DELETE CASCADE
    );

    -- Translation + fit output
    CREATE TABLE IF NOT EXISTS translations (
        chapter         INTEGER NOT NULL,
        page            INTEGER NOT NULL,
        idx             INTEGER NOT NULL,
        translated_text TEXT NOT NULL,
        font_size_px    INTEGER NOT NULL,
        line_height     REAL NOT NULL,
        overflow        INTEGER NOT NULL DEFAULT 0,
        updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (chapter, page, idx),
        FOREIGN KEY (chapter, page, idx)
            REFERENCES bubbles(chapter, page, idx) ON DELETE CASCADE
    );

    -- Audit trail
    CREATE TABLE IF NOT EXISTS translation_history (
        id              INTEGER PRIMARY KEY,
        chapter         INTEGER NOT NULL,
        page            INTEGER NOT NULL,
        idx             INTEGER NOT NULL,
        translated_text TEXT NOT NULL,
        source          TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (chapter, page, idx)
            REFERENCES bubbles(chapter, page, idx) ON DELETE CASCADE
    );

    -- Per-project glossary
    CREATE TABLE IF NOT EXISTS glossary (
        id          INTEGER PRIMARY KEY,
        source_term TEXT NOT NULL UNIQUE,
        target_term TEXT NOT NULL,
        notes       TEXT
    );

    -- Context notes (continuity between chapters)
    CREATE TABLE IF NOT EXISTS notes (
        id          INTEGER PRIMARY KEY,
        chapter     INTEGER NOT NULL,
        note_type   TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_notes_chapter ON notes(chapter);

    -- FTS indexes
    CREATE VIRTUAL TABLE IF NOT EXISTS bubbles_fts USING fts5(
        source_text, content='bubbles', content_rowid=rowid,
        tokenize='unicode61'
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS translations_fts USING fts5(
        translated_text, content='translations', content_rowid=rowid,
        tokenize='unicode61'
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS glossary_fts USING fts5(
        source_term, content='glossary', content_rowid='id',
        tokenize='unicode61'
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
        content, content='notes', content_rowid='id',
        tokenize='unicode61'
    );

    -- FTS sync triggers: glossary
    CREATE TRIGGER IF NOT EXISTS glossary_ai AFTER INSERT ON glossary BEGIN
        INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
    END;
    CREATE TRIGGER IF NOT EXISTS glossary_ad AFTER DELETE ON glossary BEGIN
        INSERT INTO glossary_fts(glossary_fts, rowid, source_term)
            VALUES('delete', old.id, old.source_term);
    END;
    CREATE TRIGGER IF NOT EXISTS glossary_au AFTER UPDATE ON glossary BEGIN
        INSERT INTO glossary_fts(glossary_fts, rowid, source_term)
            VALUES('delete', old.id, old.source_term);
        INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
    END;

    -- FTS sync triggers: notes
    CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
        INSERT INTO notes_fts(rowid, content) VALUES (new.id, new.content);
    END;
    CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
        INSERT INTO notes_fts(notes_fts, rowid, content)
            VALUES('delete', old.id, old.content);
    END;

    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;
";

// ── ProjectStore ──

impl ProjectStore {
    /// Open or create a project at the given root directory.
    /// Creates `{root}/project.db` and `{root}/masks/`, `{root}/rendered/`.
    pub fn open(root: &Path) -> Result<Self> {
        std::fs::create_dir_all(root)
            .with_context(|| format!("Failed to create project dir: {}", root.display()))?;

        let db_path = root.join("project.db");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("Failed to open project DB: {}", db_path.display()))?;
        conn.execute_batch(SCHEMA)
            .context("Failed to initialize project schema")?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            root: root.to_path_buf(),
        })
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().context("Failed to open in-memory project DB")?;
        conn.execute_batch(SCHEMA)
            .context("Failed to initialize project schema")?;
        Ok(Self {
            conn: std::sync::Mutex::new(conn),
            root: PathBuf::from("/tmp/test_project"),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Path for a bubble mask file.
    pub fn mask_path(&self, chapter: usize, page: usize, idx: usize) -> PathBuf {
        self.root
            .join("masks")
            .join(chapter.to_string())
            .join(format!("{page}_{idx}.png"))
    }

    /// Path for a rendered page image.
    pub fn rendered_path(&self, chapter: usize, page: usize) -> PathBuf {
        self.root
            .join("rendered")
            .join(chapter.to_string())
            .join(format!("page_{page:03}.png"))
    }

    // ── Pages ──

    /// Register a source image page. Returns true if detection is stale
    /// (new page or image changed).
    pub fn register_page(
        &self,
        chapter: usize,
        page: usize,
        image_hash: &str,
    ) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let existing: Option<String> = conn
            .query_row(
                "SELECT image_hash FROM pages WHERE chapter = ?1 AND page = ?2",
                params![chapter as i64, page as i64],
                |row| row.get(0),
            )
            .ok();

        match existing {
            Some(hash) if hash == image_hash => Ok(false), // up to date
            Some(_) => {
                // Image changed — clear old detection data, re-register
                conn.execute(
                    "DELETE FROM pages WHERE chapter = ?1 AND page = ?2",
                    params![chapter as i64, page as i64],
                )?;
                conn.execute(
                    "INSERT INTO pages (chapter, page, image_hash) VALUES (?1, ?2, ?3)",
                    params![chapter as i64, page as i64, image_hash],
                )?;
                Ok(true)
            }
            None => {
                conn.execute(
                    "INSERT INTO pages (chapter, page, image_hash) VALUES (?1, ?2, ?3)",
                    params![chapter as i64, page as i64, image_hash],
                )?;
                Ok(true)
            }
        }
    }

    /// Mark a page as detected.
    pub fn mark_detected(&self, chapter: usize, page: usize) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE pages SET detected_at = datetime('now')
             WHERE chapter = ?1 AND page = ?2",
            params![chapter as i64, page as i64],
        )?;
        Ok(())
    }

    /// Check if a page needs detection (not yet detected or image changed).
    pub fn needs_detect(&self, chapter: usize, page: usize) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let detected: Option<Option<String>> = conn
            .query_row(
                "SELECT detected_at FROM pages WHERE chapter = ?1 AND page = ?2",
                params![chapter as i64, page as i64],
                |row| row.get(0),
            )
            .ok();
        Ok(detected.is_none() || detected == Some(None))
    }

    // ── Bubbles (detection output) ──

    /// Save detection results for a page. Replaces any existing bubbles.
    pub fn save_bubbles(&self, chapter: usize, page: usize, bubbles: &[BubbleRow]) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM bubbles WHERE chapter = ?1 AND page = ?2",
            params![chapter as i64, page as i64],
        )?;

        let mut stmt = conn.prepare(
            "INSERT INTO bubbles (chapter, page, idx, source_text, polygon, drawable_area, det_confidence, ocr_confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;

        for b in bubbles {
            stmt.execute(params![
                b.chapter as i64,
                b.page as i64,
                b.idx as i64,
                b.source_text,
                b.polygon,
                b.drawable_area,
                b.det_confidence,
                b.ocr_confidence,
            ])?;
        }

        Ok(())
    }

    /// Get all bubbles for a page.
    pub fn get_bubbles(&self, chapter: usize, page: usize) -> Result<Vec<BubbleRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT chapter, page, idx, source_text, polygon, drawable_area, det_confidence, ocr_confidence
             FROM bubbles WHERE chapter = ?1 AND page = ?2 ORDER BY idx",
        )?;
        let rows = stmt
            .query_map(params![chapter as i64, page as i64], |row| {
                Ok(BubbleRow {
                    chapter: row.get::<_, i64>(0)? as usize,
                    page: row.get::<_, i64>(1)? as usize,
                    idx: row.get::<_, i64>(2)? as usize,
                    source_text: row.get(3)?,
                    polygon: row.get(4)?,
                    drawable_area: row.get(5)?,
                    det_confidence: row.get(6)?,
                    ocr_confidence: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Get all bubbles for a chapter.
    pub fn get_chapter_bubbles(&self, chapter: usize) -> Result<Vec<BubbleRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT chapter, page, idx, source_text, polygon, drawable_area, det_confidence, ocr_confidence
             FROM bubbles WHERE chapter = ?1 ORDER BY page, idx",
        )?;
        let rows = stmt
            .query_map(params![chapter as i64], |row| {
                Ok(BubbleRow {
                    chapter: row.get::<_, i64>(0)? as usize,
                    page: row.get::<_, i64>(1)? as usize,
                    idx: row.get::<_, i64>(2)? as usize,
                    source_text: row.get(3)?,
                    polygon: row.get(4)?,
                    drawable_area: row.get(5)?,
                    det_confidence: row.get(6)?,
                    ocr_confidence: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ── Translations ──

    /// Save translation results for a page. Replaces existing and logs history.
    pub fn save_translations(
        &self,
        _chapter: usize,
        _page: usize,
        translations: &[TranslationRow],
        source: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        for t in translations {
            conn.execute(
                "INSERT OR REPLACE INTO translations (chapter, page, idx, translated_text, font_size_px, line_height, overflow)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    t.chapter as i64,
                    t.page as i64,
                    t.idx as i64,
                    t.translated_text,
                    t.font_size_px as i64,
                    t.line_height,
                    t.overflow as i64,
                ],
            )?;

            conn.execute(
                "INSERT INTO translation_history (chapter, page, idx, translated_text, source)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    t.chapter as i64,
                    t.page as i64,
                    t.idx as i64,
                    t.translated_text,
                    source,
                ],
            )?;
        }

        Ok(())
    }

    /// Update a single bubble's translation (manual edit).
    pub fn update_translation(
        &self,
        chapter: usize,
        page: usize,
        idx: usize,
        translated_text: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE translations SET translated_text = ?4, updated_at = datetime('now')
             WHERE chapter = ?1 AND page = ?2 AND idx = ?3",
            params![chapter as i64, page as i64, idx as i64, translated_text],
        )?;
        conn.execute(
            "INSERT INTO translation_history (chapter, page, idx, translated_text, source)
             VALUES (?1, ?2, ?3, ?4, 'manual')",
            params![chapter as i64, page as i64, idx as i64, translated_text],
        )?;
        Ok(())
    }

    /// Get translations for a page.
    pub fn get_translations(&self, chapter: usize, page: usize) -> Result<Vec<TranslationRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT chapter, page, idx, translated_text, font_size_px, line_height, overflow, updated_at
             FROM translations WHERE chapter = ?1 AND page = ?2 ORDER BY idx",
        )?;
        let rows = stmt
            .query_map(params![chapter as i64, page as i64], |row| {
                Ok(TranslationRow {
                    chapter: row.get::<_, i64>(0)? as usize,
                    page: row.get::<_, i64>(1)? as usize,
                    idx: row.get::<_, i64>(2)? as usize,
                    translated_text: row.get(3)?,
                    font_size_px: row.get::<_, i64>(4)? as u32,
                    line_height: row.get(5)?,
                    overflow: row.get::<_, i64>(6)? != 0,
                    updated_at: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Check if a page has all translations (bubble count == translation count).
    pub fn is_translated(&self, chapter: usize, page: usize) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let counts: (i64, i64) = conn.query_row(
            "SELECT
                (SELECT COUNT(*) FROM bubbles WHERE chapter = ?1 AND page = ?2),
                (SELECT COUNT(*) FROM translations WHERE chapter = ?1 AND page = ?2)",
            params![chapter as i64, page as i64],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(counts.0 > 0 && counts.0 == counts.1)
    }

    /// Get the latest translation update time for a page.
    pub fn latest_translation_time(
        &self,
        chapter: usize,
        page: usize,
    ) -> Result<Option<String>> {
        let conn = self.conn.lock().unwrap();
        let ts = conn
            .query_row(
                "SELECT MAX(updated_at) FROM translations WHERE chapter = ?1 AND page = ?2",
                params![chapter as i64, page as i64],
                |row| row.get(0),
            )
            .ok();
        Ok(ts)
    }

    // ── Glossary ──

    pub fn glossary_search(&self, text: &str) -> Result<Vec<GlossaryEntry>> {
        let query = text.replace('"', "");
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT g.source_term, g.target_term, g.notes
             FROM glossary g
             JOIN glossary_fts f ON g.id = f.rowid
             WHERE glossary_fts MATCH ?1
             LIMIT 50",
        )?;
        let entries = stmt
            .query_map([&query], |row| {
                Ok(GlossaryEntry {
                    source_term: row.get(0)?,
                    target_term: row.get(1)?,
                    notes: row.get(2)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(entries)
    }

    pub fn glossary_search_batch(&self, texts: &[&str]) -> Result<Vec<GlossaryEntry>> {
        let combined = texts.join(" ");
        let mut entries = self.glossary_search(&combined)?;
        let mut seen = std::collections::HashSet::new();
        entries.retain(|e| seen.insert(e.source_term.clone()));
        Ok(entries)
    }

    pub fn glossary_upsert(
        &self,
        source_term: &str,
        target_term: &str,
        notes: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO glossary (source_term, target_term, notes) VALUES (?1, ?2, ?3)
             ON CONFLICT(source_term) DO UPDATE SET target_term = ?2, notes = ?3",
            params![source_term, target_term, notes],
        )?;
        Ok(())
    }

    /// Import glossary from TOML file.
    pub fn glossary_import_toml(&self, toml_path: &Path) -> Result<usize> {
        #[derive(serde::Deserialize)]
        struct File {
            #[serde(default)]
            terms: Vec<Term>,
        }
        #[derive(serde::Deserialize)]
        struct Term {
            source: String,
            target: String,
            notes: Option<String>,
        }

        let text = std::fs::read_to_string(toml_path)
            .with_context(|| format!("Failed to read glossary TOML: {}", toml_path.display()))?;
        let file: File = toml::from_str(&text).context("Failed to parse glossary TOML")?;

        let conn = self.conn.lock().unwrap();
        let mut count = 0;
        for term in &file.terms {
            conn.execute(
                "INSERT INTO glossary (source_term, target_term, notes) VALUES (?1, ?2, ?3)
                 ON CONFLICT(source_term) DO UPDATE SET target_term = ?2, notes = ?3",
                params![term.source, term.target, term.notes],
            )?;
            count += 1;
        }
        conn.execute("INSERT INTO glossary_fts(glossary_fts) VALUES('rebuild')", [])?;
        Ok(count)
    }

    // ── Context notes ──

    pub fn add_note(&self, chapter: usize, note_type: &str, content: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO notes (chapter, note_type, content) VALUES (?1, ?2, ?3)",
            params![chapter as i64, note_type, content],
        )?;
        Ok(())
    }

    pub fn get_notes_before(&self, chapter: usize) -> Result<Vec<NoteRow>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, chapter, note_type, content FROM notes
             WHERE chapter < ?1 ORDER BY chapter DESC",
        )?;
        let rows = stmt
            .query_map(params![chapter as i64], |row| {
                Ok(NoteRow {
                    id: row.get(0)?,
                    chapter: row.get::<_, i64>(1)? as usize,
                    note_type: row.get(2)?,
                    content: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// FTS search across translations and notes.
    pub fn search_context(&self, query: &str, chapter: usize) -> Result<Vec<String>> {
        let query = query.replace('"', "");
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();

        let mut results = Vec::new();

        // Search translations
        {
            let mut stmt = conn.prepare(
                "SELECT t.translated_text, b.source_text, t.chapter, t.page
                 FROM translations_fts tf
                 JOIN translations t ON t.rowid = tf.rowid
                 JOIN bubbles b ON b.chapter = t.chapter AND b.page = t.page AND b.idx = t.idx
                 WHERE translations_fts MATCH ?1 AND t.chapter <= ?2
                 ORDER BY rank LIMIT 20",
            )?;
            let hits: Vec<String> = stmt
                .query_map(params![&query, chapter as i64], |row| {
                    let translated: String = row.get(0)?;
                    let source: String = row.get(1)?;
                    let ch: i64 = row.get(2)?;
                    let pg: i64 = row.get(3)?;
                    Ok(format!("[ch{ch} p{pg}] {source} → {translated}"))
                })?
                .filter_map(|r| r.ok())
                .collect();
            results.extend(hits);
        }

        // Search notes
        {
            let mut stmt = conn.prepare(
                "SELECT n.content, n.note_type, n.chapter
                 FROM notes_fts nf
                 JOIN notes n ON n.id = nf.rowid
                 WHERE notes_fts MATCH ?1 AND n.chapter <= ?2
                 ORDER BY rank LIMIT 10",
            )?;
            let hits: Vec<String> = stmt
                .query_map(params![&query, chapter as i64], |row| {
                    let content: String = row.get(0)?;
                    let note_type: String = row.get(1)?;
                    let ch: i64 = row.get(2)?;
                    Ok(format!("[ch{ch} {note_type}] {content}"))
                })?
                .filter_map(|r| r.ok())
                .collect();
            results.extend(hits);
        }

        Ok(results)
    }

    /// Check if any translations or notes exist.
    pub fn has_data(&self) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let has: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM translations) OR EXISTS(SELECT 1 FROM notes)",
            [],
            |row| row.get(0),
        )?;
        Ok(has)
    }

    /// Get all source->translated pairs for a chapter (for context agent).
    pub fn get_chapter_pairs(&self, chapter: usize) -> Result<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT b.source_text, t.translated_text
             FROM translations t
             JOIN bubbles b ON b.chapter = t.chapter AND b.page = t.page AND b.idx = t.idx
             WHERE t.chapter = ?1
             ORDER BY t.page, t.idx",
        )?;
        let rows = stmt
            .query_map(params![chapter as i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Batch FTS search for context agent (AND-first, OR fallback).
    pub fn batch_search_context(
        &self,
        queries: &[String],
        scope: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }
        let conn = self.conn.lock().unwrap();
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let search_t = scope == "all" || scope == "translations";
        let search_n = scope == "all" || scope == "notes";

        for query in queries {
            let query = query.trim().replace('"', "");
            if query.is_empty() {
                continue;
            }

            if search_t {
                let mut stmt = conn.prepare(
                    "SELECT b.source_text, t.translated_text, t.chapter, t.page
                     FROM translations_fts tf
                     JOIN translations t ON t.rowid = tf.rowid
                     JOIN bubbles b ON b.chapter = t.chapter AND b.page = t.page AND b.idx = t.idx
                     WHERE translations_fts MATCH ?1
                     ORDER BY rank LIMIT ?2",
                )?;
                let hits: Vec<String> = stmt
                    .query_map(params![&query, limit as i64], |row| {
                        let source: String = row.get(0)?;
                        let translated: String = row.get(1)?;
                        let ch: i64 = row.get(2)?;
                        let pg: i64 = row.get(3)?;
                        Ok(format!("[Ch{ch} p{pg}] {source} -> {translated}"))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                for h in hits {
                    if seen.insert(h.clone()) {
                        results.push(h);
                    }
                }
            }

            if search_n {
                let mut stmt = conn.prepare(
                    "SELECT n.content, n.note_type, n.chapter
                     FROM notes_fts nf
                     JOIN notes n ON n.id = nf.rowid
                     WHERE notes_fts MATCH ?1
                     ORDER BY rank LIMIT ?2",
                )?;
                let hits: Vec<String> = stmt
                    .query_map(params![&query, limit as i64], |row| {
                        let content: String = row.get(0)?;
                        let note_type: String = row.get(1)?;
                        let ch: i64 = row.get(2)?;
                        Ok(format!("[Ch{ch} {note_type}] {content}"))
                    })?
                    .filter_map(|r| r.ok())
                    .collect();
                for h in hits {
                    if seen.insert(h.clone()) {
                        results.push(h);
                    }
                }
            }
        }

        results.truncate(limit);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_registration() -> Result<()> {
        let store = ProjectStore::open_in_memory()?;

        // First registration — needs detect
        assert!(store.register_page(1, 0, "abc123")?);
        assert!(store.needs_detect(1, 0)?);

        // Mark detected
        store.mark_detected(1, 0)?;
        assert!(!store.needs_detect(1, 0)?);

        // Same hash — no change needed
        assert!(!store.register_page(1, 0, "abc123")?);

        // Different hash — needs re-detect
        assert!(store.register_page(1, 0, "def456")?);
        assert!(store.needs_detect(1, 0)?);

        Ok(())
    }

    #[test]
    fn test_bubbles_crud() -> Result<()> {
        let store = ProjectStore::open_in_memory()?;
        store.register_page(1, 0, "hash")?;

        let bubbles = vec![
            BubbleRow {
                chapter: 1, page: 0, idx: 0,
                source_text: "hello".into(),
                polygon: "[[0,0],[100,0],[100,50],[0,50]]".into(),
                drawable_area: r#"{"x":5,"y":5,"w":90,"h":40}"#.into(),
                det_confidence: 0.95, ocr_confidence: 0.88,
            },
            BubbleRow {
                chapter: 1, page: 0, idx: 1,
                source_text: "world".into(),
                polygon: "[[0,60],[100,60],[100,110],[0,110]]".into(),
                drawable_area: r#"{"x":5,"y":65,"w":90,"h":40}"#.into(),
                det_confidence: 0.90, ocr_confidence: 0.75,
            },
        ];

        store.save_bubbles(1, 0, &bubbles)?;
        let loaded = store.get_bubbles(1, 0)?;
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].source_text, "hello");
        assert_eq!(loaded[1].source_text, "world");

        Ok(())
    }

    #[test]
    fn test_translations_and_history() -> Result<()> {
        let store = ProjectStore::open_in_memory()?;
        store.register_page(1, 0, "hash")?;
        store.save_bubbles(1, 0, &[BubbleRow {
            chapter: 1, page: 0, idx: 0,
            source_text: "hello".into(),
            polygon: "[]".into(),
            drawable_area: "{}".into(),
            det_confidence: 0.9, ocr_confidence: 0.9,
        }])?;

        // Auto translation
        store.save_translations(1, 0, &[TranslationRow {
            chapter: 1, page: 0, idx: 0,
            translated_text: "xin chào".into(),
            font_size_px: 24, line_height: 1.3, overflow: false,
            updated_at: String::new(),
        }], "auto")?;

        assert!(store.is_translated(1, 0)?);

        // Manual edit
        store.update_translation(1, 0, 0, "chào bạn")?;
        let trans = store.get_translations(1, 0)?;
        assert_eq!(trans[0].translated_text, "chào bạn");

        Ok(())
    }

    #[test]
    fn test_glossary() -> Result<()> {
        let store = ProjectStore::open_in_memory()?;
        store.glossary_upsert("선배", "tiền bối", Some("senior"))?;
        store.glossary_upsert("후배", "hậu bối", None)?;

        let results = store.glossary_search("선배")?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target_term, "tiền bối");

        // Upsert updates existing
        store.glossary_upsert("선배", "senpai", None)?;
        let results = store.glossary_search("선배")?;
        assert_eq!(results[0].target_term, "senpai");

        Ok(())
    }

    #[test]
    fn test_notes() -> Result<()> {
        let store = ProjectStore::open_in_memory()?;
        store.add_note(1, "character", "Min-jun is the protagonist")?;
        store.add_note(2, "relationship", "Min-jun and Seo-yeon are siblings")?;

        let notes = store.get_notes_before(3)?;
        assert_eq!(notes.len(), 2);

        let notes = store.get_notes_before(2)?;
        assert_eq!(notes.len(), 1);

        Ok(())
    }
}
