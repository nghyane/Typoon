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
}

#[derive(Debug, Clone)]
pub struct NoteMatch {
    pub chapter_index: usize,
    pub note_type: String,
    pub content: String,
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

            CREATE VIRTUAL TABLE IF NOT EXISTS translations_fts USING fts5(
                source_text,
                translated_text,
                content='translations',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS translations_ai AFTER INSERT ON translations BEGIN
                INSERT INTO translations_fts(rowid, source_text, translated_text)
                    VALUES (new.id, new.source_text, new.translated_text);
            END;
            CREATE TRIGGER IF NOT EXISTS translations_ad AFTER DELETE ON translations BEGIN
                INSERT INTO translations_fts(translations_fts, rowid, source_text, translated_text)
                    VALUES('delete', old.id, old.source_text, old.translated_text);
            END;

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

            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                content,
                content='chapter_notes',
                content_rowid='id',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON chapter_notes BEGIN
                INSERT INTO notes_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON chapter_notes BEGIN
                INSERT INTO notes_fts(notes_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
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

    /// Search translations using FTS5. Returns matches with project/chapter metadata.
    pub fn search_translations(
        &self,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<TranslationMatch>> {
        let query = query.replace('"', "");
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT t.chapter_index, t.page_index, t.bubble_id, t.source_text, t.translated_text
             FROM translations t
             JOIN translations_fts f ON t.id = f.rowid
             WHERE translations_fts MATCH ?1 AND t.project_id = ?2
             LIMIT ?3",
        )?;

        let matches = stmt
            .query_map((&query, project_id, limit as i64), |row| {
                Ok(TranslationMatch {
                    chapter_index: row.get::<_, i64>(0)? as usize,
                    page_index: row.get::<_, i64>(1)? as usize,
                    bubble_id: row.get(2)?,
                    source_text: row.get(3)?,
                    translated_text: row.get(4)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(matches)
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
            (project_id, chapter_index as i64, note_type, content),
        )?;

        Ok(())
    }

    /// Search notes using FTS5.
    pub fn search_notes(
        &self,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<NoteMatch>> {
        let query = query.replace('"', "");
        if query.trim().is_empty() {
            return Ok(vec![]);
        }

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT n.chapter_index, n.note_type, n.content
             FROM chapter_notes n
             JOIN notes_fts f ON n.id = f.rowid
             WHERE notes_fts MATCH ?1 AND n.project_id = ?2
             LIMIT ?3",
        )?;

        let matches = stmt
            .query_map((&query, project_id, limit as i64), |row| {
                Ok(NoteMatch {
                    chapter_index: row.get::<_, i64>(0)? as usize,
                    note_type: row.get(1)?,
                    content: row.get(2)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(matches)
    }

    /// Get all notes for a project up to (but not including) the given chapter.
    pub fn get_notes_before(
        &self,
        project_id: &str,
        chapter_index: usize,
    ) -> Result<Vec<NoteMatch>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT chapter_index, note_type, content
             FROM chapter_notes
             WHERE project_id = ?1 AND chapter_index < ?2
             ORDER BY chapter_index, id",
        )?;

        let notes = stmt
            .query_map((project_id, chapter_index as i64), |row| {
                Ok(NoteMatch {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_translation(page: usize, bubble: &str, source: &str, translated: &str) -> ChapterTranslation {
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

        let results = store.search_translations("Xin", "project-1", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].translated_text, "Xin chào");
        assert_eq!(results[0].chapter_index, 1);
        assert_eq!(results[0].page_index, 0);
        assert_eq!(results[0].bubble_id, "b1");
    }

    #[test]
    fn test_save_chapter_idempotent() {
        let store = ContextStore::open_in_memory().unwrap();

        let translations = vec![
            make_translation(0, "b1", "こんにちは", "Xin chào"),
        ];

        store.save_chapter("project-1", 1, &translations).unwrap();
        store.save_chapter("project-1", 1, &translations).unwrap();

        let all = store.get_chapter_translations("project-1", 1).unwrap();
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_add_and_search_notes() {
        let store = ContextStore::open_in_memory().unwrap();

        store.add_note("project-1", 1, "character", "Tanaka is the protagonist").unwrap();
        store.add_note("project-1", 1, "event", "Battle at the castle gate").unwrap();
        store.add_note("project-1", 2, "relationship", "Tanaka and Yuki are siblings").unwrap();

        let results = store.search_notes("Tanaka", "project-1", 10).unwrap();
        assert_eq!(results.len(), 2);

        let results = store.search_notes("castle", "project-1", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].note_type, "event");
    }

    #[test]
    fn test_get_chapter_translations() {
        let store = ContextStore::open_in_memory().unwrap();

        let ch1 = vec![
            make_translation(0, "b1", "テスト", "Kiểm tra"),
            make_translation(1, "b2", "データ", "Dữ liệu"),
        ];
        let ch2 = vec![
            make_translation(0, "b1", "別章", "Chương khác"),
        ];

        store.save_chapter("project-1", 1, &ch1).unwrap();
        store.save_chapter("project-1", 2, &ch2).unwrap();

        let result = store.get_chapter_translations("project-1", 1).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].source_text, "テスト");
        assert_eq!(result[1].source_text, "データ");

        let result = store.get_chapter_translations("project-1", 2).unwrap();
        assert_eq!(result.len(), 1);
    }
}
