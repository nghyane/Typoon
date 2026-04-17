"""SQLite storage adapter."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from typoon.types import Bubble

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            INTEGER PRIMARY KEY,
    source_url    TEXT,
    title         TEXT,
    source_lang   TEXT NOT NULL DEFAULT 'en',
    target_lang   TEXT NOT NULL DEFAULT 'vi',
    auto_update   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chapters (
    id            INTEGER PRIMARY KEY,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    idx           REAL NOT NULL,
    local_path    TEXT,
    source_url    TEXT,
    source_name   TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    retry_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, idx)
);

CREATE TABLE IF NOT EXISTS glossary (
    id            INTEGER PRIMARY KEY,
    project_id    INTEGER NOT NULL REFERENCES projects(id),
    source_term   TEXT NOT NULL,
    target_term   TEXT NOT NULL,
    notes         TEXT,
    UNIQUE(project_id, source_term)
);

CREATE TABLE IF NOT EXISTS translations (
    project_id    INTEGER NOT NULL,
    chapter       REAL NOT NULL,
    page          INTEGER NOT NULL,
    idx           INTEGER NOT NULL,
    source_text   TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    polygon       TEXT,
    font_size_px  INTEGER,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, chapter, page, idx)
);

CREATE TABLE IF NOT EXISTS knowledge_snapshots (
    project_id    INTEGER NOT NULL,
    chapter       REAL NOT NULL,
    snapshot      TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, chapter)
);

CREATE TABLE IF NOT EXISTS notes (
    id            INTEGER PRIMARY KEY,
    project_id    INTEGER NOT NULL,
    chapter       REAL NOT NULL,
    note_type     TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_notes_project_ch ON notes(project_id, chapter);

-- FTS for glossary search
CREATE VIRTUAL TABLE IF NOT EXISTS glossary_fts USING fts5(
    source_term, content='glossary', content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS glossary_ai AFTER INSERT ON glossary BEGIN
    INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
END;
CREATE TRIGGER IF NOT EXISTS glossary_ad AFTER DELETE ON glossary BEGIN
    INSERT INTO glossary_fts(glossary_fts, rowid, source_term) VALUES('delete', old.id, old.source_term);
END;
CREATE TRIGGER IF NOT EXISTS glossary_au AFTER UPDATE ON glossary BEGIN
    INSERT INTO glossary_fts(glossary_fts, rowid, source_term) VALUES('delete', old.id, old.source_term);
    INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
END;

-- FTS for translation search
CREATE VIRTUAL TABLE IF NOT EXISTS translations_fts USING fts5(
    source_text, translated_text, content='translations', content_rowid=rowid,
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS translations_ai AFTER INSERT ON translations BEGIN
    INSERT INTO translations_fts(rowid, source_text, translated_text) VALUES (new.rowid, new.source_text, new.translated_text);
END;
CREATE TRIGGER IF NOT EXISTS translations_ad AFTER DELETE ON translations BEGIN
    INSERT INTO translations_fts(translations_fts, rowid, source_text, translated_text) VALUES('delete', old.rowid, old.source_text, old.translated_text);
END;

-- FTS for notes search
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    content, content='notes', content_rowid='id',
    tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
"""


class SqliteStore:
    """SQLite implementation of Store port."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    @staticmethod
    async def open(path: str | Path) -> SqliteStore:
        db = await aiosqlite.connect(str(path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(_SCHEMA)
        return SqliteStore(db)

    @staticmethod
    async def open_memory() -> SqliteStore:
        return await SqliteStore.open(":memory:")

    async def close(self) -> None:
        await self._db.close()

    # ── Projects ─────────────────────────────────────────────────

    async def add_project(
        self, title: str, source_lang: str = "en", target_lang: str = "vi",
        source_url: str | None = None, auto_update: bool = False,
    ) -> int:
        cur = await self._db.execute(
            "INSERT INTO projects (title, source_lang, target_lang, source_url, auto_update) VALUES (?,?,?,?,?)",
            (title, source_lang, target_lang, source_url, int(auto_update)),
        )
        await self._db.commit()
        return cur.lastrowid  # type: ignore

    async def get_project(self, project_id: int) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_project_by_title(self, title: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE title=?", (title,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_project_by_url(self, source_url: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE source_url=?", (source_url,))
        row = await cur.fetchone()
        return dict(row) if row else None

    # ── Chapters ─────────────────────────────────────────────────

    async def add_chapter(
        self, project_id: int, idx: float,
        local_path: str | None = None, source_url: str | None = None,
        source_name: str | None = None,
    ) -> int:
        cur = await self._db.execute(
            "INSERT OR IGNORE INTO chapters (project_id, idx, local_path, source_url, source_name) VALUES (?,?,?,?,?)",
            (project_id, idx, local_path, source_url, source_name),
        )
        await self._db.commit()
        return cur.lastrowid  # type: ignore

    async def set_chapter_status(self, project_id: int, idx: float, status: str) -> None:
        await self._db.execute(
            "UPDATE chapters SET status=? WHERE project_id=? AND idx=?",
            (status, project_id, idx),
        )
        await self._db.commit()

    async def get_chapters(self, project_id: int) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM chapters WHERE project_id=? ORDER BY idx", (project_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def get_chapter_status(self, project_id: int, idx: float) -> str | None:
        cur = await self._db.execute(
            "SELECT status FROM chapters WHERE project_id=? AND idx=?",
            (project_id, idx),
        )
        row = await cur.fetchone()
        return row["status"] if row else None

    async def get_chapter_retry_count(self, project_id: int, idx: float) -> int:
        cur = await self._db.execute(
            "SELECT retry_count FROM chapters WHERE project_id=? AND idx=?",
            (project_id, idx),
        )
        row = await cur.fetchone()
        return row["retry_count"] if row else 0

    async def increment_retry_count(self, project_id: int, idx: float) -> None:
        await self._db.execute(
            "UPDATE chapters SET retry_count = retry_count + 1 WHERE project_id=? AND idx=?",
            (project_id, idx),
        )
        await self._db.commit()

    async def delete_chapter_data(self, project_id: int, idx: float) -> None:
        """Remove translations + knowledge + notes for a chapter (for re-translate)."""
        await self._db.execute(
            "DELETE FROM translations WHERE project_id=? AND chapter=?", (project_id, idx),
        )
        await self._db.execute(
            "DELETE FROM knowledge_snapshots WHERE project_id=? AND chapter=?", (project_id, idx),
        )
        await self._db.execute(
            "DELETE FROM notes WHERE project_id=? AND chapter=?", (project_id, idx),
        )
        await self._db.commit()

    # ── Glossary (Store port) ────────────────────────────────────

    async def get_glossary(self, project_id: int) -> dict[str, str]:
        cur = await self._db.execute(
            "SELECT source_term, target_term FROM glossary WHERE project_id=?", (project_id,),
        )
        return {r["source_term"]: r["target_term"] for r in await cur.fetchall()}

    async def glossary_upsert(
        self, project_id: int, source_term: str, target_term: str, notes: str | None = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO glossary (project_id, source_term, target_term, notes) VALUES (?,?,?,?) "
            "ON CONFLICT(project_id, source_term) DO UPDATE SET target_term=excluded.target_term, notes=excluded.notes",
            (project_id, source_term, target_term, notes),
        )
        await self._db.commit()

    async def glossary_search(self, project_id: int, query: str) -> list[dict]:
        cur = await self._db.execute(
            "SELECT g.source_term, g.target_term, g.notes FROM glossary_fts f "
            "JOIN glossary g ON g.id = f.rowid WHERE f.source_term MATCH ? AND g.project_id=? LIMIT 10",
            (query, project_id),
        )
        return [dict(r) for r in await cur.fetchall()]

    # ── Knowledge (Store port) ───────────────────────────────────

    async def get_knowledge(self, project_id: int, before_chapter: float) -> str | None:
        cur = await self._db.execute(
            "SELECT snapshot FROM knowledge_snapshots WHERE project_id=? AND chapter<? ORDER BY chapter DESC LIMIT 1",
            (project_id, before_chapter),
        )
        row = await cur.fetchone()
        return row["snapshot"] if row else None

    async def save_knowledge(self, project_id: int, chapter: float, snapshot: str) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO knowledge_snapshots (project_id, chapter, snapshot) VALUES (?,?,?)",
            (project_id, chapter, snapshot),
        )
        await self._db.commit()

    # ── Translations (Store port) ────────────────────────────────

    async def save_translations(self, project_id: int, chapter: float, bubbles: list[Bubble]) -> None:
        rows = [
            (project_id, chapter, b.page_index, b.idx, b.source_text,
             b.translated_text or "", json.dumps(b.polygon), b.font_size)
            for b in bubbles
        ]
        await self._db.executemany(
            "INSERT OR REPLACE INTO translations "
            "(project_id, chapter, page, idx, source_text, translated_text, polygon, font_size_px) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        await self._db.commit()

    async def get_chapter_translations(self, project_id: int, chapter: float) -> list[dict]:
        cur = await self._db.execute(
            "SELECT page, idx, source_text, translated_text FROM translations "
            "WHERE project_id=? AND chapter=? ORDER BY page, idx",
            (project_id, chapter),
        )
        return [dict(r) for r in await cur.fetchall()]

    # ── Notes ────────────────────────────────────────────────────

    async def add_note(self, project_id: int, chapter: float, note_type: str, content: str) -> None:
        await self._db.execute(
            "INSERT INTO notes (project_id, chapter, note_type, content) VALUES (?,?,?,?)",
            (project_id, chapter, note_type, content),
        )
        await self._db.commit()

    # ── Context search ───────────────────────────────────────────

    async def search_context(self, project_id: int, queries: list[str], scope: str = "all", limit: int = 12) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()

        for query in queries:
            if scope in ("all", "translations"):
                cur = await self._db.execute(
                    "SELECT t.source_text, t.translated_text, t.chapter, t.page "
                    "FROM translations_fts f "
                    "JOIN translations t ON t.rowid = f.rowid "
                    "WHERE translations_fts MATCH ? AND t.project_id=? "
                    "ORDER BY rank LIMIT ?",
                    (query, project_id, limit),
                )
                for r in await cur.fetchall():
                    h = f"[Ch{r[2]} p{r[3]}] {r[0]} → {r[1]}"
                    if h not in seen:
                        seen.add(h)
                        results.append(h)

            if scope in ("all", "notes"):
                cur = await self._db.execute(
                    "SELECT n.content, n.note_type, n.chapter "
                    "FROM notes_fts f "
                    "JOIN notes n ON n.id = f.rowid "
                    "WHERE notes_fts MATCH ? AND n.project_id=? "
                    "ORDER BY rank LIMIT ?",
                    (query, project_id, limit),
                )
                for r in await cur.fetchall():
                    h = f"[Ch{r[2]} {r[1]}] {r[0]}"
                    if h not in seen:
                        seen.add(h)
                        results.append(h)

        return results[:limit]

    async def get_chapter_pairs(self, project_id: int, chapter: float) -> list[tuple[str, str]]:
        cur = await self._db.execute(
            "SELECT source_text, translated_text FROM translations "
            "WHERE project_id=? AND chapter=? ORDER BY page, idx",
            (project_id, chapter),
        )
        return [(r[0], r[1]) for r in await cur.fetchall()]
