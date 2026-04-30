"""SQLite storage adapter."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from typoon.domain.bubble import Bubble

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
    key           TEXT NOT NULL,
    source_text   TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'ok',
    polygon       TEXT,
    font_size_px  INTEGER,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, chapter, page, idx),
    UNIQUE(project_id, chapter, key)
);

CREATE TABLE IF NOT EXISTS chapter_briefs (
    project_id    INTEGER NOT NULL,
    chapter       REAL NOT NULL,
    brief_json    TEXT NOT NULL,
    summary       TEXT,
    terms_text    TEXT,
    facts_text    TEXT,
    rules_text    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (project_id, chapter)
);

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
        """Remove translations + chapter brief for a chapter (for re-translate)."""
        await self._db.execute(
            "DELETE FROM translations WHERE project_id=? AND chapter=?", (project_id, idx),
        )
        await self._db.execute(
            "DELETE FROM chapter_briefs WHERE project_id=? AND chapter=?", (project_id, idx),
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
        safe = _fts_escape(query)
        if not safe:
            return []
        cur = await self._db.execute(
            "SELECT g.source_term, g.target_term, g.notes FROM glossary_fts f "
            "JOIN glossary g ON g.id = f.rowid WHERE f.source_term MATCH ? AND g.project_id=? LIMIT 10",
            (safe, project_id),
        )
        return [dict(r) for r in await cur.fetchall()]

    # ── Chapter briefs / context ─────────────────────────────────

    async def save_chapter_brief(self, project_id: int, chapter: float, brief: dict) -> None:
        brief_json = json.dumps(brief, ensure_ascii=False)
        summary = str(brief.get("summary", ""))
        terms = brief.get("glossary", {}) or {}
        terms_text = "\n".join(f"{k} -> {v}" for k, v in terms.items())
        facts_text = "\n".join(str(x) for x in brief.get("facts", []) or [])
        rules = list(brief.get("rules", []) or [])
        rules_text = "\n".join(str(x) for x in rules)
        await self._db.execute(
            "INSERT OR REPLACE INTO chapter_briefs "
            "(project_id, chapter, brief_json, summary, terms_text, facts_text, rules_text, updated_at) "
            "VALUES (?,?,?,?,?,?,?,datetime('now'))",
            (project_id, chapter, brief_json, summary, terms_text, facts_text, rules_text),
        )
        await self._db.commit()

    async def get_chapter_brief(self, project_id: int, chapter: float) -> dict | None:
        cur = await self._db.execute(
            "SELECT chapter, brief_json, summary, terms_text, facts_text, rules_text "
            "FROM chapter_briefs WHERE project_id=? AND chapter=?",
            (project_id, chapter),
        )
        row = await cur.fetchone()
        if not row:
            return None
        out = dict(row)
        out["brief"] = json.loads(out.pop("brief_json"))
        return out

    async def get_recent_chapter_briefs(
        self, project_id: int, before_chapter: float, limit: int = 3,
    ) -> list[dict]:
        cur = await self._db.execute(
            "SELECT chapter, brief_json, summary, terms_text, facts_text, rules_text "
            "FROM chapter_briefs WHERE project_id=? AND chapter<? "
            "ORDER BY chapter DESC LIMIT ?",
            (project_id, before_chapter, limit),
        )
        rows = []
        for row in await cur.fetchall():
            out = dict(row)
            out["brief"] = json.loads(out.pop("brief_json"))
            rows.append(out)
        return rows

    async def search_briefs(self, project_id: int, queries: list[str], limit: int = 10) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for query in queries:
            like = f"%{query}%"
            cur = await self._db.execute(
                "SELECT chapter, summary, terms_text, facts_text, rules_text FROM chapter_briefs "
                "WHERE project_id=? AND (summary LIKE ? OR terms_text LIKE ? OR facts_text LIKE ? OR rules_text LIKE ?) "
                "ORDER BY chapter DESC LIMIT ?",
                (project_id, like, like, like, like, limit),
            )
            for r in await cur.fetchall():
                text = "\n".join(
                    str(r[k] or "") for k in ("summary", "terms_text", "facts_text", "rules_text")
                ).strip()
                hit = f"[Ch{r['chapter']} brief] {text}"
                if text and hit not in seen:
                    seen.add(hit)
                    results.append(hit)
        return results[:limit]

    # ── Translations (Store port) ────────────────────────────────

    async def save_translations(self, project_id: int, chapter: float, bubbles: list[Bubble]) -> None:
        rows = [
            (project_id, chapter, b.page_index, b.idx, b.translation_key or b.id,
             b.source_text, b.translated_text or "", b.translation_status,
             json.dumps(b.polygon), b.font_size)
            for b in bubbles
        ]
        await self._db.executemany(
            "INSERT OR REPLACE INTO translations "
            "(project_id, chapter, page, idx, key, source_text, translated_text, status, polygon, font_size_px) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        await self._db.commit()

    async def get_chapter_translations(self, project_id: int, chapter: float) -> list[dict]:
        cur = await self._db.execute(
            "SELECT page, idx, key, source_text, translated_text, status FROM translations "
            "WHERE project_id=? AND chapter=? ORDER BY page, idx",
            (project_id, chapter),
        )
        return [dict(r) for r in await cur.fetchall()]

    # ── Context search ───────────────────────────────────────────

    async def search_context(self, project_id: int, queries: list[str], scope: str = "all", limit: int = 12) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()

        for query in queries:
            safe = _fts_escape(query)
            if not safe:
                continue
            if scope in ("all", "translations"):
                cur = await self._db.execute(
                    "SELECT t.source_text, t.translated_text, t.chapter, t.page "
                    "FROM translations_fts f "
                    "JOIN translations t ON t.rowid = f.rowid "
                    "WHERE translations_fts MATCH ? AND t.project_id=? "
                    "ORDER BY rank LIMIT ?",
                    (safe, project_id, limit),
                )
                for r in await cur.fetchall():
                    h = f"[Ch{r[2]} p{r[3]}] {r[0]} → {r[1]}"
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

    async def update_bubble(self, project_id: int, chapter: float, page: int, idx: int, text: str) -> None:
        await self._db.execute(
            "UPDATE translations SET translated_text=? "
            "WHERE project_id=? AND chapter=? AND page=? AND idx=?",
            (text, project_id, chapter, page, idx))
        await self._db.commit()

    async def list_projects(self) -> list[dict]:
        cur = await self._db.execute("SELECT * FROM projects ORDER BY id DESC")
        return [dict(r) for r in await cur.fetchall()]


import re as _re

_FTS_SPECIAL = _re.compile(r"[\"'\(\)\*\:\^\-\+\{\}\[\]~]")


def _fts_escape(query: str) -> str:
    """Sanitize a query string for FTS5 MATCH."""
    clean = _FTS_SPECIAL.sub(" ", query).strip()
    words = clean.split()
    if not words:
        return ""
    return " ".join(f'"{w}"' for w in words)
