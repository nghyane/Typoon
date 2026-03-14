use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::Connection;

#[derive(Debug, Clone)]
pub struct ChapterTranslation {
    pub page_index: usize,
    pub bubble_id: String,
    pub source_text: String,
    pub translated_text: String,
    pub source_lang: String,
    pub target_lang: String,
}

#[derive(Debug, Clone)]
pub struct TranslationMatch {
    pub chapter_index: usize,
    pub page_index: usize,
    pub bubble_id: String,
    pub source_text: String,
    pub translated_text: String,
    pub rank: f64,
}

#[derive(Debug, Clone)]
pub struct NoteMatch {
    pub chapter_index: usize,
    pub note_type: String,
    pub content: String,
    pub rank: f64,
}

/// Lightweight note record for direct lookups (no search scoring).
#[derive(Debug, Clone)]
pub struct NoteRecord {
    pub chapter_index: usize,
    pub note_type: String,
    pub content: String,
}

/// Scope for batch search.
#[derive(Debug, Clone, Copy, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchScope {
    Translations,
    Notes,
    #[default]
    All,
}

/// Unified search hit across translations and notes.
#[derive(Debug, Clone)]
pub struct ContextHit {
    pub kind: ContextHitKind,
    pub chapter_index: usize,
    pub summary: String,
    pub rank: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContextHitKind {
    Translation,
    Note,
}

/// SQLite-backed context store for cross-chapter translation context.
/// Wrapped in Mutex for thread safety (rusqlite::Connection is !Sync).
pub struct ContextStore {
    conn: std::sync::Mutex<Connection>,
}

impl ContextStore {
    /// Open (or create) a context database at the given path.
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open context DB: {}", db_path.display()))?;

        Self::init_schema(&conn)?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
        })
    }

    #[cfg(test)]
    fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("Failed to open in-memory context DB")?;

        Self::init_schema(&conn)?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
        })
    }

    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                chapter_index INTEGER NOT NULL,
                page_index INTEGER NOT NULL,
                bubble_id TEXT NOT NULL,
                source_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_translations_project_chapter
                ON translations(project_id, chapter_index);

            CREATE TABLE IF NOT EXISTS chapter_notes (
                id INTEGER PRIMARY KEY,
                project_id TEXT NOT NULL,
                chapter_index INTEGER NOT NULL,
                note_type TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_notes_project_chapter
                ON chapter_notes(project_id, chapter_index);

            CREATE VIRTUAL TABLE IF NOT EXISTS translations_fts USING fts5(
                source_text, translated_text, content='translations', content_rowid='id'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                content, content='chapter_notes', content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS translations_ai AFTER INSERT ON translations BEGIN
                INSERT INTO translations_fts(rowid, source_text, translated_text)
                VALUES (new.id, new.source_text, new.translated_text);
            END;

            CREATE TRIGGER IF NOT EXISTS translations_ad AFTER DELETE ON translations BEGIN
                INSERT INTO translations_fts(translations_fts, rowid, source_text, translated_text)
                VALUES ('delete', old.id, old.source_text, old.translated_text);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON chapter_notes BEGIN
                INSERT INTO notes_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON chapter_notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;",
        )
        .context("Failed to initialize context schema")?;

        Ok(())
    }

    /// Save a full chapter's translations. Deletes any existing translations for this
    /// (project_id, chapter_index) first (idempotent re-runs).
    pub fn save_chapter(
        &self,
        project_id: &str,
        chapter_index: usize,
        translations: &[ChapterTranslation],
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "DELETE FROM translations WHERE project_id = ?1 AND chapter_index = ?2",
            (project_id, chapter_index as i64),
        )?;

        let mut stmt = conn.prepare(
            "INSERT INTO translations (project_id, chapter_index, page_index, bubble_id, source_text, translated_text, source_lang, target_lang)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;

        for t in translations {
            stmt.execute((
                project_id,
                chapter_index as i64,
                t.page_index as i64,
                &t.bubble_id,
                &t.source_text,
                &t.translated_text,
                &t.source_lang,
                &t.target_lang,
            ))?;
        }

        Ok(())
    }

    /// Check if any data exists for a project.
    pub fn has_data(&self, project_id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let has_translations: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM translations WHERE project_id = ?1)",
            [project_id],
            |row| row.get(0),
        )?;
        if has_translations {
            return Ok(true);
        }
        let has_notes: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM chapter_notes WHERE project_id = ?1)",
            [project_id],
            |row| row.get(0),
        )?;
        Ok(has_notes)
    }

    /// Search translations using FTS5 full-text search.
    /// Uses AND-first strategy; falls back to OR if AND yields no results.
    pub fn search_translations(
        &self,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<TranslationMatch>> {
        if query.is_empty() {
            return Ok(vec![]);
        }

        let conn = self.conn.lock().unwrap();

        // Try AND first
        let results = Self::run_translation_fts(&conn, &fts_and_query(query), project_id, limit)?;
        if !results.is_empty() {
            return Ok(results);
        }

        // Fallback to OR
        Self::run_translation_fts(&conn, &fts_or_query(query), project_id, limit)
    }

    fn run_translation_fts(
        conn: &Connection,
        fts_query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<TranslationMatch>> {
        let mut stmt = conn.prepare(
            "SELECT t.chapter_index, t.page_index, t.bubble_id, t.source_text, t.translated_text, translations_fts.rank
             FROM translations_fts
             JOIN translations t ON t.id = translations_fts.rowid
             WHERE translations_fts MATCH ?1 AND t.project_id = ?2
             ORDER BY translations_fts.rank
             LIMIT ?3",
        )?;

        let rows = stmt.query_map((fts_query, project_id, limit as i64), |row| {
            Ok(TranslationMatch {
                chapter_index: row.get::<_, i64>(0)? as usize,
                page_index: row.get::<_, i64>(1)? as usize,
                bubble_id: row.get(2)?,
                source_text: row.get(3)?,
                translated_text: row.get(4)?,
                rank: row.get(5)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Add a chapter note (LLM-generated observation).
    pub fn add_note(
        &self,
        project_id: &str,
        chapter_index: usize,
        note_type: &str,
        content: &str,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chapter_notes (project_id, chapter_index, note_type, content)
             VALUES (?1, ?2, ?3, ?4)",
            (
                project_id,
                chapter_index as i64,
                note_type,
                content,
            ),
        )?;

        Ok(())
    }

    /// Search notes using FTS5. AND-first with OR fallback.
    pub fn search_notes(
        &self,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<NoteMatch>> {
        if query.is_empty() {
            return Ok(vec![]);
        }

        let conn = self.conn.lock().unwrap();

        let results = Self::run_notes_fts(&conn, &fts_and_query(query), project_id, limit)?;
        if !results.is_empty() {
            return Ok(results);
        }

        Self::run_notes_fts(&conn, &fts_or_query(query), project_id, limit)
    }

    fn run_notes_fts(
        conn: &Connection,
        fts_query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<NoteMatch>> {
        let mut stmt = conn.prepare(
            "SELECT n.chapter_index, n.note_type, n.content, notes_fts.rank
             FROM notes_fts
             JOIN chapter_notes n ON n.id = notes_fts.rowid
             WHERE notes_fts MATCH ?1 AND n.project_id = ?2
             ORDER BY notes_fts.rank
             LIMIT ?3",
        )?;

        let rows = stmt.query_map((fts_query, project_id, limit as i64), |row| {
            Ok(NoteMatch {
                chapter_index: row.get::<_, i64>(0)? as usize,
                note_type: row.get(1)?,
                content: row.get(2)?,
                rank: row.get(3)?,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get all notes for a project up to (but not including) the given chapter.
    pub fn get_notes_before(
        &self,
        project_id: &str,
        chapter_index: usize,
    ) -> Result<Vec<NoteRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT chapter_index, note_type, content
             FROM chapter_notes
             WHERE project_id = ?1 AND chapter_index < ?2
             ORDER BY chapter_index, id",
        )?;

        let notes = stmt
            .query_map((project_id, chapter_index as i64), |row| {
                Ok(NoteRecord {
                    chapter_index: row.get::<_, i64>(0)? as usize,
                    note_type: row.get(1)?,
                    content: row.get(2)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(notes)
    }

    /// Get all translations for a specific chapter.
    pub fn get_chapter_translations(
        &self,
        project_id: &str,
        chapter_index: usize,
    ) -> Result<Vec<ChapterTranslation>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT page_index, bubble_id, source_text, translated_text, source_lang, target_lang
             FROM translations
             WHERE project_id = ?1 AND chapter_index = ?2
             ORDER BY page_index, bubble_id",
        )?;

        let translations = stmt
            .query_map((project_id, chapter_index as i64), |row| {
                Ok(ChapterTranslation {
                    page_index: row.get::<_, i64>(0)? as usize,
                    bubble_id: row.get(1)?,
                    source_text: row.get(2)?,
                    translated_text: row.get(3)?,
                    source_lang: row.get(4)?,
                    target_lang: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(translations)
    }

    /// Batch search: multiple queries, deduplicated, across translations and/or notes.
    /// Returns up to `limit` hits sorted by relevance.
    pub fn batch_search(
        &self,
        project_id: &str,
        queries: &[String],
        scope: SearchScope,
        limit: usize,
    ) -> Result<Vec<ContextHit>> {
        if queries.is_empty() || !self.has_data(project_id)? {
            return Ok(vec![]);
        }

        let conn = self.conn.lock().unwrap();
        let mut hits: Vec<ContextHit> = Vec::new();
        let mut seen_translations = std::collections::HashSet::new();
        let mut seen_notes = std::collections::HashSet::new();

        let search_t = matches!(scope, SearchScope::All | SearchScope::Translations);
        let search_n = matches!(scope, SearchScope::All | SearchScope::Notes);
        let per_query_limit = limit;

        for query in queries {
            let query = query.trim();
            if query.is_empty() {
                continue;
            }

            if search_t {
                // AND-first, OR fallback
                let mut results =
                    Self::run_translation_fts(&conn, &fts_and_query(query), project_id, per_query_limit)?;
                if results.is_empty() {
                    results =
                        Self::run_translation_fts(&conn, &fts_or_query(query), project_id, per_query_limit)?;
                }
                for m in results {
                    let key = (m.chapter_index, m.page_index, m.bubble_id.clone());
                    if seen_translations.insert(key) {
                        hits.push(ContextHit {
                            kind: ContextHitKind::Translation,
                            chapter_index: m.chapter_index,
                            summary: format!(
                                "[p{} {}] {} → {}",
                                m.page_index, m.bubble_id, m.source_text, m.translated_text
                            ),
                            rank: m.rank,
                        });
                    }
                }
            }

            if search_n {
                let mut results =
                    Self::run_notes_fts(&conn, &fts_and_query(query), project_id, per_query_limit)?;
                if results.is_empty() {
                    results =
                        Self::run_notes_fts(&conn, &fts_or_query(query), project_id, per_query_limit)?;
                }
                for m in results {
                    let key = (m.chapter_index, m.note_type.clone(), m.content.clone());
                    if seen_notes.insert(key) {
                        hits.push(ContextHit {
                            kind: ContextHitKind::Note,
                            chapter_index: m.chapter_index,
                            summary: format!("[{}] {}", m.note_type, m.content),
                            rank: m.rank,
                        });
                    }
                }
            }
        }

        // Sort by rank (BM25: lower = more relevant)
        hits.sort_by(|a, b| a.rank.partial_cmp(&b.rank).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        Ok(hits)
    }
}

// ── FTS5 query builders ──

/// AND-first: all tokens must appear (FTS5 implicit AND).
fn fts_and_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{}\"", w.replace('"', "")))
        .collect();
    if tokens.is_empty() {
        "\"\"".to_string()
    } else {
        tokens.join(" ")
    }
}

/// OR fallback: any token matches.
fn fts_or_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split_whitespace()
        .filter(|w| !w.is_empty())
        .map(|w| format!("\"{}\"", w.replace('"', "")))
        .collect();
    if tokens.is_empty() {
        "\"\"".to_string()
    } else {
        tokens.join(" OR ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_translation(
        page: usize,
        bubble: &str,
        source: &str,
        translated: &str,
    ) -> ChapterTranslation {
        ChapterTranslation {
            page_index: page,
            bubble_id: bubble.to_string(),
            source_text: source.to_string(),
            translated_text: translated.to_string(),
            source_lang: "ja".to_string(),
            target_lang: "vi".to_string(),
        }
    }

    #[test]
    fn test_open_creates_db() {
        let store = ContextStore::open_in_memory().unwrap();
        drop(store);
    }

    #[test]
    fn test_save_and_search_translations() {
        let store = ContextStore::open_in_memory().unwrap();

        let translations = vec![
            make_translation(0, "b1", "こんにちは", "Xin chào"),
            make_translation(0, "b2", "さようなら", "Tạm biệt"),
            make_translation(1, "b3", "ありがとう", "Cảm ơn"),
        ];

        store.save_chapter("project-1", 1, &translations).unwrap();

        // Search for a term that appears in a translation
        let results = store
            .search_translations("Xin chào", "project-1", 10)
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].bubble_id, "b1");
    }

    #[test]
    fn test_save_chapter_idempotent() {
        let store = ContextStore::open_in_memory().unwrap();

        let translations = vec![make_translation(0, "b1", "こんにちは", "Xin chào")];

        store.save_chapter("project-1", 1, &translations).unwrap();
        store.save_chapter("project-1", 1, &translations).unwrap();

        let all = store.get_chapter_translations("project-1", 1).unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_add_and_search_notes() {
        let store = ContextStore::open_in_memory().unwrap();

        store
            .add_note("project-1", 1, "character", "Tanaka is the protagonist")
            .unwrap();
        store
            .add_note("project-1", 1, "event", "Battle at the castle gate")
            .unwrap();
        store
            .add_note("project-1", 2, "relationship", "Tanaka and Yuki are siblings")
            .unwrap();

        // Search for notes mentioning Tanaka
        let results = store.search_notes("Tanaka", "project-1", 10).unwrap();
        assert!(!results.is_empty());
        // Should find notes containing "Tanaka"
        assert!(results.iter().any(|r| r.content.contains("Tanaka")));
    }

    #[test]
    fn test_get_chapter_translations() {
        let store = ContextStore::open_in_memory().unwrap();

        let ch1 = vec![
            make_translation(0, "b1", "テスト", "Kiểm tra"),
            make_translation(1, "b2", "データ", "Dữ liệu"),
        ];
        let ch2 = vec![make_translation(0, "b1", "別章", "Chương khác")];

        store.save_chapter("project-1", 1, &ch1).unwrap();
        store.save_chapter("project-1", 2, &ch2).unwrap();

        let result = store.get_chapter_translations("project-1", 1).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].source_text, "テスト");
        assert_eq!(result[1].source_text, "データ");

        let result = store.get_chapter_translations("project-1", 2).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_search_no_results_for_unrelated_query() {
        let store = ContextStore::open_in_memory().unwrap();

        let translations = vec![make_translation(0, "b1", "テスト", "Test")];

        store.save_chapter("project-1", 1, &translations).unwrap();

        // Search for something completely unrelated
        let results = store
            .search_translations("dinosaur spaceship", "project-1", 10)
            .unwrap();
        assert!(results.is_empty());
    }
}
