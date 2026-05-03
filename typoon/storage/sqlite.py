"""SQLite storage — identity, knowledge, worker coordination."""

from __future__ import annotations

import json
import re as _re
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    source_url   TEXT,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chapters (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx          REAL NOT NULL,
    source_url   TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, idx)
);

-- Worker coordination — row exists = pending/running, DELETE = done
CREATE TABLE IF NOT EXISTS tasks (
    chapter_id   INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL CHECK(stage IN ('prepare','scan','translate','render')),
    claimed_by   TEXT,
    claimed_at   TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    PRIMARY KEY (chapter_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_tasks_claim ON tasks(stage, claimed_by, claimed_at);

-- OCR text — structured, searchable
CREATE TABLE IF NOT EXISTS bubbles (
    chapter_id   INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    bubble_idx   INTEGER NOT NULL,
    source_text  TEXT NOT NULL,
    confidence   REAL NOT NULL,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

-- Translations — knowledge, reusable across reruns
CREATE TABLE IF NOT EXISTS translations (
    chapter_id      INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    translated_text TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK(kind IN ('dialogue','sfx','skip')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chapter_id, page_index, bubble_idx),
    FOREIGN KEY (chapter_id, page_index, bubble_idx)
        REFERENCES bubbles(chapter_id, page_index, bubble_idx) ON DELETE CASCADE
);

-- Chapter context briefs
CREATE TABLE IF NOT EXISTS chapter_briefs (
    chapter_id   INTEGER PRIMARY KEY REFERENCES chapters(id) ON DELETE CASCADE,
    brief_json   TEXT NOT NULL,
    summary      TEXT,
    terms_text   TEXT,
    facts_text   TEXT,
    rules_text   TEXT,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Project-level glossary
CREATE TABLE IF NOT EXISTS glossary (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    notes        TEXT,
    UNIQUE(project_id, source_term)
);

-- FTS
CREATE VIRTUAL TABLE IF NOT EXISTS bubbles_fts USING fts5(
    source_text, content='bubbles', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS bubbles_ai AFTER INSERT ON bubbles BEGIN
    INSERT INTO bubbles_fts(rowid, source_text) VALUES (new.rowid, new.source_text);
END;
CREATE TRIGGER IF NOT EXISTS bubbles_ad AFTER DELETE ON bubbles BEGIN
    INSERT INTO bubbles_fts(bubbles_fts, rowid, source_text) VALUES ('delete', old.rowid, old.source_text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS translations_fts USING fts5(
    source_text, translated_text,
    content='translations', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS translations_ai AFTER INSERT ON translations BEGIN
    INSERT INTO translations_fts(rowid, source_text, translated_text)
    SELECT new.rowid, b.source_text, new.translated_text
    FROM   bubbles b
    WHERE  b.chapter_id=new.chapter_id AND b.page_index=new.page_index AND b.bubble_idx=new.bubble_idx;
END;
CREATE TRIGGER IF NOT EXISTS translations_ad AFTER DELETE ON translations BEGIN
    INSERT INTO translations_fts(translations_fts, rowid, source_text, translated_text)
    VALUES ('delete', old.rowid, '', '');
END;

CREATE VIRTUAL TABLE IF NOT EXISTS briefs_fts USING fts5(
    summary, terms_text, facts_text, rules_text,
    content='chapter_briefs', content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS briefs_ai AFTER INSERT ON chapter_briefs BEGIN
    INSERT INTO briefs_fts(rowid, summary, terms_text, facts_text, rules_text)
    VALUES (new.rowid, new.summary, new.terms_text, new.facts_text, new.rules_text);
END;
CREATE TRIGGER IF NOT EXISTS briefs_ad AFTER DELETE ON chapter_briefs BEGIN
    INSERT INTO briefs_fts(briefs_fts, rowid, summary, terms_text, facts_text, rules_text)
    VALUES ('delete', old.rowid, old.summary, old.terms_text, old.facts_text, old.rules_text);
END;
CREATE TRIGGER IF NOT EXISTS briefs_au AFTER UPDATE ON chapter_briefs BEGIN
    INSERT INTO briefs_fts(briefs_fts, rowid, summary, terms_text, facts_text, rules_text)
    VALUES ('delete', old.rowid, old.summary, old.terms_text, old.facts_text, old.rules_text);
    INSERT INTO briefs_fts(rowid, summary, terms_text, facts_text, rules_text)
    VALUES (new.rowid, new.summary, new.terms_text, new.facts_text, new.rules_text);
END;

CREATE VIRTUAL TABLE IF NOT EXISTS glossary_fts USING fts5(
    source_term, content='glossary', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS glossary_ai AFTER INSERT ON glossary BEGIN
    INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
END;
CREATE TRIGGER IF NOT EXISTS glossary_ad AFTER DELETE ON glossary BEGIN
    INSERT INTO glossary_fts(glossary_fts, rowid, source_term) VALUES ('delete', old.id, old.source_term);
END;
CREATE TRIGGER IF NOT EXISTS glossary_au AFTER UPDATE ON glossary BEGIN
    INSERT INTO glossary_fts(glossary_fts, rowid, source_term) VALUES ('delete', old.id, old.source_term);
    INSERT INTO glossary_fts(rowid, source_term) VALUES (new.id, new.source_term);
END;
"""


class SqliteStore:
    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    @staticmethod
    async def open(path: str | Path) -> "SqliteStore":
        db = await aiosqlite.connect(str(path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(_SCHEMA)
        return SqliteStore(db)

    @staticmethod
    async def open_memory() -> "SqliteStore":
        return await SqliteStore.open(":memory:")

    async def close(self) -> None:
        await self._db.close()

    # ── Projects ──────────────────────────────────────────────────

    async def get_or_create_project(
        self,
        slug: str,
        title: str,
        source_lang: str,
        target_lang: str,
        source_url: str | None = None,
    ) -> int:
        cur = await self._db.execute("SELECT id FROM projects WHERE slug=?", (slug,))
        row = await cur.fetchone()
        if row:
            return row["id"]
        if source_url:
            cur = await self._db.execute("SELECT id FROM projects WHERE source_url=?", (source_url,))
            row = await cur.fetchone()
            if row:
                return row["id"]
        return await self._add_project(slug, title, source_lang, target_lang, source_url)

    async def _add_project(
        self,
        slug: str,
        title: str,
        source_lang: str,
        target_lang: str,
        source_url: str | None = None,
    ) -> int:
        cur = await self._db.execute(
            "INSERT INTO projects (slug, title, source_lang, target_lang, source_url) VALUES (?,?,?,?,?)",
            (slug, title, source_lang, target_lang, source_url),
        )
        await self._db.commit()
        return cur.lastrowid  # type: ignore

    async def get_project(self, project_id: int) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_projects(self) -> list[dict]:
        cur = await self._db.execute("SELECT * FROM projects ORDER BY id DESC")
        return [dict(r) for r in await cur.fetchall()]

    # ── Chapters ──────────────────────────────────────────────────

    async def get_or_create_chapter(
        self,
        project_id: int,
        idx: float,
        source_url: str | None = None,
    ) -> int:
        """Return chapter_id (existing or new)."""
        cur = await self._db.execute(
            "SELECT id FROM chapters WHERE project_id=? AND idx=?", (project_id, idx)
        )
        row = await cur.fetchone()
        if row:
            return row["id"]
        cur = await self._db.execute(
            "INSERT INTO chapters (project_id, idx, source_url) VALUES (?,?,?)",
            (project_id, idx, source_url),
        )
        await self._db.commit()
        return cur.lastrowid  # type: ignore

    async def get_chapter(self, chapter_id: int) -> dict | None:
        cur = await self._db.execute("SELECT * FROM chapters WHERE id=?", (chapter_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_chapter_by_idx(self, project_id: int, idx: float) -> dict | None:
        cur = await self._db.execute(
            "SELECT * FROM chapters WHERE project_id=? AND idx=?", (project_id, idx)
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_all_chapters(self, project_id: int) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM chapters WHERE project_id=? ORDER BY idx", (project_id,)
        )
        return [dict(r) for r in await cur.fetchall()]

    # ── Tasks ─────────────────────────────────────────────────────

    async def enqueue(self, chapter_id: int, stage: str) -> None:
        """Add a task if not already present."""
        await self._db.execute(
            "INSERT OR IGNORE INTO tasks (chapter_id, stage) VALUES (?,?)",
            (chapter_id, stage),
        )
        await self._db.commit()

    async def enqueue_many(self, chapter_id: int, stages: list[str]) -> None:
        await self._db.executemany(
            "INSERT OR IGNORE INTO tasks (chapter_id, stage) VALUES (?,?)",
            [(chapter_id, s) for s in stages],
        )
        await self._db.commit()

    async def claim_task(self, stage: str, worker_id: str) -> int | None:
        """Atomically claim one pending task. Returns chapter_id or None."""
        # Two-step: find candidate then update — aiosqlite doesn't support RETURNING
        # with open cursor + commit in same transaction cleanly.
        async with self._db.execute(
            """
            SELECT rowid, chapter_id FROM tasks
            WHERE  stage = ?
              AND  (claimed_by IS NULL
                    OR claimed_at < datetime('now', '-10 minutes'))
            ORDER  BY chapter_id
            LIMIT  1
            """,
            (stage,),
        ) as cur:
            row = await cur.fetchone()

        if row is None:
            return None

        await self._db.execute(
            """
            UPDATE tasks
            SET    claimed_by = ?,
                   claimed_at = datetime('now')
            WHERE  rowid = ?
              AND  (claimed_by IS NULL
                    OR claimed_at < datetime('now', '-10 minutes'))
            """,
            (worker_id, row["rowid"]),
        )
        await self._db.commit()

        # Verify we actually claimed it (another worker could have raced)
        async with self._db.execute(
            "SELECT chapter_id FROM tasks WHERE rowid=? AND claimed_by=?",
            (row["rowid"], worker_id),
        ) as cur2:
            claimed = await cur2.fetchone()

        return claimed["chapter_id"] if claimed else None

    async def complete_task(self, chapter_id: int, stage: str) -> None:
        """Delete task on success."""
        await self._db.execute(
            "DELETE FROM tasks WHERE chapter_id=? AND stage=?",
            (chapter_id, stage),
        )
        await self._db.commit()

    async def fail_task(self, chapter_id: int, stage: str, error: str) -> None:
        """Release claim, increment attempts, record error."""
        await self._db.execute(
            """
            UPDATE tasks
            SET    claimed_by = NULL,
                   claimed_at = NULL,
                   attempts   = attempts + 1,
                   last_error = ?
            WHERE  chapter_id=? AND stage=?
            """,
            (error, chapter_id, stage),
        )
        await self._db.commit()

    async def get_tasks(self, chapter_id: int) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM tasks WHERE chapter_id=?", (chapter_id,)
        )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_tasks_from(self, chapter_id: int, stage: str) -> None:
        """Remove tasks from a given stage onward (for redo)."""
        order = ["prepare", "scan", "translate", "render"]
        if stage not in order:
            return
        stages_to_delete = order[order.index(stage):]
        await self._db.executemany(
            "DELETE FROM tasks WHERE chapter_id=? AND stage=?",
            [(chapter_id, s) for s in stages_to_delete],
        )
        await self._db.commit()

    # ── Bubbles ───────────────────────────────────────────────────

    async def save_bubbles(
        self,
        chapter_id: int,
        bubbles: list[dict],  # [{page_index, bubble_idx, source_text, confidence}]
    ) -> None:
        """Replace all bubbles for a chapter (idempotent rescan)."""
        await self._db.execute("DELETE FROM bubbles WHERE chapter_id=?", (chapter_id,))
        await self._db.executemany(
            "INSERT INTO bubbles (chapter_id, page_index, bubble_idx, source_text, confidence) "
            "VALUES (?,?,?,?,?)",
            [(chapter_id, b["page_index"], b["bubble_idx"], b["source_text"], b["confidence"])
             for b in bubbles],
        )
        await self._db.commit()

    async def get_bubbles(self, chapter_id: int) -> list[dict]:
        cur = await self._db.execute(
            "SELECT page_index, bubble_idx, source_text, confidence "
            "FROM bubbles WHERE chapter_id=? ORDER BY page_index, bubble_idx",
            (chapter_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def delete_chapter_data(self, chapter_id: int) -> None:
        """Delete all derived data for a chapter — keeps identity row."""
        for table in ("bubbles", "translations", "chapter_briefs", "tasks"):
            await self._db.execute(f"DELETE FROM {table} WHERE chapter_id=?", (chapter_id,))
        await self._db.commit()

    async def has_bubbles(self, chapter_id: int) -> bool:
        cur = await self._db.execute(
            "SELECT 1 FROM bubbles WHERE chapter_id=? LIMIT 1", (chapter_id,)
        )
        return await cur.fetchone() is not None

    # ── Translations ──────────────────────────────────────────────

    async def save_translations(
        self,
        chapter_id: int,
        records: list[dict],  # [{page_index, bubble_idx, translated_text, kind}]
    ) -> None:
        """Replace all translations for a chapter."""
        await self._db.execute(
            "DELETE FROM translations WHERE chapter_id=?", (chapter_id,)
        )
        await self._db.executemany(
            "INSERT INTO translations (chapter_id, page_index, bubble_idx, translated_text, kind) "
            "VALUES (?,?,?,?,?)",
            [(chapter_id, r["page_index"], r["bubble_idx"], r["translated_text"], r["kind"])
             for r in records],
        )
        await self._db.commit()

    async def get_translations(self, chapter_id: int) -> dict[tuple[int, int], dict]:
        """Return {(page_index, bubble_idx): {translated_text, kind}}."""
        cur = await self._db.execute(
            "SELECT page_index, bubble_idx, translated_text, kind "
            "FROM translations WHERE chapter_id=? ORDER BY page_index, bubble_idx",
            (chapter_id,),
        )
        return {
            (r["page_index"], r["bubble_idx"]): dict(r)
            for r in await cur.fetchall()
        }

    async def has_translations(self, chapter_id: int) -> bool:
        cur = await self._db.execute(
            "SELECT 1 FROM translations WHERE chapter_id=? LIMIT 1", (chapter_id,)
        )
        return await cur.fetchone() is not None

    # ── Glossary ──────────────────────────────────────────────────

    async def get_glossary(self, project_id: int) -> dict[str, str]:
        cur = await self._db.execute(
            "SELECT source_term, target_term FROM glossary WHERE project_id=?", (project_id,)
        )
        return {r["source_term"]: r["target_term"] for r in await cur.fetchall()}

    async def glossary_search(self, project_id: int, query: str) -> list[dict]:
        safe = _fts_escape(query)
        if not safe:
            return []
        cur = await self._db.execute(
            "SELECT g.source_term, g.target_term, g.notes "
            "FROM glossary_fts f JOIN glossary g ON g.id=f.rowid "
            "WHERE f.source_term MATCH ? AND g.project_id=? LIMIT 10",
            (safe, project_id),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def upsert_glossary_terms(self, project_id: int, terms: dict[str, str]) -> None:
        if not terms:
            return
        await self._db.executemany(
            "INSERT OR REPLACE INTO glossary (project_id, source_term, target_term) VALUES (?,?,?)",
            [(project_id, src, tgt) for src, tgt in terms.items()],
        )
        await self._db.commit()

    # ── Chapter briefs ────────────────────────────────────────────

    async def save_chapter_brief(self, chapter_id: int, brief: dict) -> None:
        terms   = brief.get("glossary", {}) or {}
        style   = brief.get("style_notes", brief.get("rules", [])) or []
        address = brief.get("address", []) or []
        address_text = "\n".join(
            f"{a.get('speaker','')} → {a.get('listener','')}: "
            f"{a.get('self_ref','')}/{a.get('other_ref','')}"
            for a in address
        )
        await self._db.execute(
            "INSERT OR REPLACE INTO chapter_briefs "
            "(chapter_id, brief_json, summary, terms_text, facts_text, rules_text, updated_at) "
            "VALUES (?,?,?,?,?,?,datetime('now'))",
            (
                chapter_id,
                json.dumps(brief, ensure_ascii=False),
                str(brief.get("summary", "")),
                "\n".join(f"{k} -> {v}" for k, v in terms.items()),
                "\n".join(str(x) for x in brief.get("facts", []) or []),
                address_text + "\n" + "\n".join(str(x) for x in style),
            ),
        )
        await self._db.commit()

        # Propagate glossary terms to project level
        if terms:
            cur = await self._db.execute(
                "SELECT project_id FROM chapters WHERE id=?", (chapter_id,)
            )
            row = await cur.fetchone()
            if row:
                await self.upsert_glossary_terms(row["project_id"], terms)

    async def get_chapter_brief(self, chapter_id: int) -> dict | None:
        cur = await self._db.execute(
            "SELECT brief_json, summary, terms_text, facts_text, rules_text "
            "FROM chapter_briefs WHERE chapter_id=?",
            (chapter_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        out = dict(row)
        out["brief"] = json.loads(out.pop("brief_json"))
        return out

    async def get_recent_chapter_briefs(
        self,
        project_id: int,
        before_chapter_idx: float,
        limit: int = 3,
    ) -> list[dict]:
        """Return briefs for recent chapters before the given idx, newest first."""
        cur = await self._db.execute(
            "SELECT c.idx, cb.brief_json, cb.summary, cb.terms_text, cb.facts_text, cb.rules_text "
            "FROM chapter_briefs cb "
            "JOIN chapters c ON c.id = cb.chapter_id "
            "WHERE c.project_id=? AND c.idx<? "
            "ORDER BY c.idx DESC LIMIT ?",
            (project_id, before_chapter_idx, limit),
        )
        rows = []
        for row in await cur.fetchall():
            out = dict(row)
            out["chapter"] = out.pop("idx")
            out["brief"] = json.loads(out.pop("brief_json"))
            rows.append(out)
        return rows

    async def search_briefs(
        self,
        project_id: int,
        queries: list[str],
        limit: int = 10,
        *,
        before_chapter_idx: float | None = None,
    ) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for query in queries:
            safe = _fts_escape(query)
            if not safe:
                continue
            chapter_filter = "AND c.idx < ?" if before_chapter_idx is not None else ""
            params: list = [safe, project_id]
            if before_chapter_idx is not None:
                params.append(before_chapter_idx)
            params.append(limit)
            cur = await self._db.execute(
                "SELECT c.idx, cb.summary, cb.terms_text, cb.facts_text, cb.rules_text "
                "FROM briefs_fts f "
                "JOIN chapter_briefs cb ON cb.rowid = f.rowid "
                "JOIN chapters c ON c.id = cb.chapter_id "
                f"WHERE briefs_fts MATCH ? AND c.project_id=? {chapter_filter} "
                "ORDER BY rank LIMIT ?",
                tuple(params),
            )
            for r in await cur.fetchall():
                text = "\n".join(
                    str(r[k] or "") for k in ("summary", "terms_text", "facts_text", "rules_text")
                ).strip()
                hit = f"[Ch{r['idx']} brief] {text}"
                if text and hit not in seen:
                    seen.add(hit)
                    results.append(hit)
        return results[:limit]

    async def search_context(
        self,
        project_id: int,
        queries: list[str],
        scope: str = "all",
        limit: int = 12,
    ) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for query in queries:
            safe = _fts_escape(query)
            if not safe:
                continue
            if scope in ("all", "translations"):
                cur = await self._db.execute(
                    "SELECT b.source_text, t.translated_text, c.idx, t.page_index "
                    "FROM translations_fts f "
                    "JOIN translations t ON t.rowid = f.rowid "
                    "JOIN bubbles b ON b.chapter_id=t.chapter_id "
                    "  AND b.page_index=t.page_index AND b.bubble_idx=t.bubble_idx "
                    "JOIN chapters c ON c.id = t.chapter_id "
                    "WHERE translations_fts MATCH ? AND c.project_id=? ORDER BY rank LIMIT ?",
                    (safe, project_id, limit),
                )
                for r in await cur.fetchall():
                    h = f"[Ch{r['idx']} p{r['page_index']}] {r['source_text']} → {r['translated_text']}"
                    if h not in seen:
                        seen.add(h)
                        results.append(h)
        return results[:limit]


# ── Helpers ───────────────────────────────────────────────────────────

_FTS_SPECIAL = _re.compile(r"[\"'\(\)\*\:\^\-\+\{\}\[\]~]")


def _fts_escape(query: str) -> str:
    clean = _FTS_SPECIAL.sub(" ", query).strip()
    words = clean.split()
    return " ".join(f'"{w}"' for w in words) if words else ""
