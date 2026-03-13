use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::Deserialize;

/// A single glossary entry: source term → target translation.
#[derive(Debug, Clone)]
pub struct GlossaryEntry {
    pub source_term: String,
    pub target_term: String,
    pub notes: Option<String>,
}

/// SQLite-backed glossary with FTS5 full-text search.
/// Wrapped in Mutex for thread safety (rusqlite::Connection is !Sync).
pub struct Glossary {
    conn: std::sync::Mutex<Connection>,
}

/// TOML file format for importing glossary.
#[derive(Deserialize)]
struct GlossaryFile {
    #[serde(default)]
    terms: Vec<TermEntry>,
}

#[derive(Deserialize)]
struct TermEntry {
    source: String,
    target: String,
    notes: Option<String>,
}

impl Glossary {
    /// Open (or create) a glossary database at the given path.
    pub fn open(db_path: &Path) -> Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(db_path)
            .with_context(|| format!("Failed to open glossary DB: {}", db_path.display()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY,
                source_term TEXT NOT NULL,
                target_term TEXT NOT NULL,
                notes TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS glossary_fts USING fts5(
                source_term,
                content='glossary',
                content_rowid='id',
                tokenize='unicode61'
            );
            -- Triggers to keep FTS in sync
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
            END;",
        )
        .context("Failed to initialize glossary schema")?;

        Ok(Self { conn: std::sync::Mutex::new(conn) })
    }

    /// Import terms from a TOML file. Upserts: existing source_term gets updated.
    pub fn import_toml(&self, toml_path: &Path) -> Result<usize> {
        let text = std::fs::read_to_string(toml_path)
            .with_context(|| format!("Failed to read glossary TOML: {}", toml_path.display()))?;
        let file: GlossaryFile =
            toml::from_str(&text).context("Failed to parse glossary TOML")?;

        let conn = self.conn.lock().unwrap();
        let mut count = 0;
        for term in &file.terms {
            conn.execute(
                "INSERT INTO glossary (source_term, target_term, notes) VALUES (?1, ?2, ?3)
                 ON CONFLICT DO NOTHING",
                (&term.source, &term.target, &term.notes),
            )?;
            count += 1;
        }

        // Rebuild FTS index after bulk import
        conn.execute("INSERT INTO glossary_fts(glossary_fts) VALUES('rebuild')", [])?;

        tracing::info!("Imported {count} glossary terms from {}", toml_path.display());
        Ok(count)
    }

    /// Search glossary using FTS5 MATCH against OCR text.
    /// Extracts individual tokens from `ocr_text` and finds matching glossary entries.
    pub fn search(&self, ocr_text: &str) -> Result<Vec<GlossaryEntry>> {
        if ocr_text.trim().is_empty() {
            return Ok(vec![]);
        }

        // Use the raw OCR text as an OR query — FTS5 will match any token
        // Escape double-quotes to prevent FTS syntax errors
        let query = ocr_text.replace('"', "");
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

    /// Insert or update a glossary term. If source_term already exists, update target + notes.
    pub fn upsert(&self, source_term: &str, target_term: &str, notes: Option<&str>) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let existing: Option<i64> = conn
            .query_row(
                "SELECT id FROM glossary WHERE source_term = ?1",
                [source_term],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            conn.execute(
                "UPDATE glossary SET target_term = ?1, notes = ?2 WHERE id = ?3",
                (target_term, notes, id),
            )?;
        } else {
            conn.execute(
                "INSERT INTO glossary (source_term, target_term, notes) VALUES (?1, ?2, ?3)",
                (source_term, target_term, notes),
            )?;
        }

        Ok(())
    }

    /// Search with multiple OCR texts (e.g., all bubbles in a chapter).
    /// Deduplicates results by source_term.
    pub fn search_batch(&self, texts: &[&str]) -> Result<Vec<GlossaryEntry>> {
        let combined = texts.join(" ");
        let mut entries = self.search(&combined)?;

        // Deduplicate by source_term
        let mut seen = std::collections::HashSet::new();
        entries.retain(|e| seen.insert(e.source_term.clone()));

        Ok(entries)
    }
}
