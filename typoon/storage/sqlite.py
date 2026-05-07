"""SQLite storage — identity, knowledge, worker coordination."""

from __future__ import annotations

import json
import re as _re
from pathlib import Path

import aiosqlite


async def _migrate_chapters(db: aiosqlite.Connection) -> None:
    """Idempotent ALTER TABLE for chapters columns added after initial release."""
    cur = await db.execute("PRAGMA table_info(chapters)")
    cols = {row["name"] for row in await cur.fetchall()}
    additions = [
        ("title",          "TEXT"),
        ("rendered",       "INTEGER NOT NULL DEFAULT 0"),
        ("page_count",     "INTEGER NOT NULL DEFAULT 0"),
        # SQLite forbids non-constant DEFAULT in ADD COLUMN; backfill below.
        ("updated_at",     "TEXT"),
    ]
    for name, ddl in additions:
        if name not in cols:
            await db.execute(f"ALTER TABLE chapters ADD COLUMN {name} {ddl}")
    await db.execute(
        "UPDATE chapters SET updated_at=created_at WHERE updated_at IS NULL"
    )
    # Drop legacy columns from earlier RFC-004 drafts. Idempotent.
    for legacy in ("prepared_key", "render_key", "render_job_id", "render_state"):
        if legacy in cols:
            await db.execute(f"ALTER TABLE chapters DROP COLUMN {legacy}")
    await db.commit()


async def _migrate_projects(db: aiosqlite.Connection) -> None:
    """Idempotent ALTER TABLE for projects metadata columns."""
    cur = await db.execute("PRAGMA table_info(projects)")
    cols = {row["name"] for row in await cur.fetchall()}
    additions = [
        ("description", "TEXT"),
        ("cover_path",  "TEXT"),
        ("updated_at",  "TEXT"),  # see _migrate_chapters note
        ("settings",    "TEXT"),  # JSON blob; per-project overrides
    ]
    for name, ddl in additions:
        if name not in cols:
            await db.execute(f"ALTER TABLE projects ADD COLUMN {name} {ddl}")
    await db.execute(
        "UPDATE projects SET updated_at=created_at WHERE updated_at IS NULL"
    )
    await db.commit()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id           INTEGER PRIMARY KEY,
    slug         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    description  TEXT,
    cover_path   TEXT,
    source_url   TEXT,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    settings     TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chapters (
    id           INTEGER PRIMARY KEY,
    project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx          REAL NOT NULL,
    title        TEXT,
    source_url   TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(project_id, idx)
);

-- Identity (Phase 1: Discord OAuth, designed polymorphic so other providers
-- can be added without migration)
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY,
    display_name  TEXT NOT NULL,
    avatar_url    TEXT,
    email         TEXT,
    tier          TEXT NOT NULL DEFAULT 'member',  -- member | admin
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS identities (
    id           INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider     TEXT NOT NULL,                    -- 'discord' for Phase 1
    external_id  TEXT NOT NULL,                    -- discord snowflake
    metadata     TEXT,                             -- JSON: raw provider payload
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(provider, external_id)
);
CREATE INDEX IF NOT EXISTS idx_identities_user ON identities(user_id);

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

-- Bubble geometry — replaces the legacy scan.npz file.
-- One row per bubble; polygon/box payloads are JSON arrays.
CREATE TABLE IF NOT EXISTS bubble_geometry (
    chapter_id   INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    bubble_idx   INTEGER NOT NULL,
    polygon      TEXT NOT NULL,
    fit_box      TEXT NOT NULL,
    erase_box    TEXT NOT NULL,
    text_box     TEXT NOT NULL,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

CREATE TABLE IF NOT EXISTS page_geometry (
    chapter_id   INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index   INTEGER NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    PRIMARY KEY (chapter_id, page_index)
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

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    data       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Triggers run AFTER tables exist and AFTER migrations have backfilled
# updated_at. Otherwise the cascade triggers fire during backfill and stomp
# project.updated_at with `now()`.
_TRIGGERS = """
-- updated_at maintenance.
-- Touch chapters.updated_at on any column change except updated_at itself
-- (recursive_triggers=off in default mode prevents the loop, but the WHEN
-- clause makes intent explicit and keeps things safe if it gets toggled).
CREATE TRIGGER IF NOT EXISTS chapters_touch_updated_at
AFTER UPDATE ON chapters
WHEN NEW.updated_at IS OLD.updated_at
BEGIN
    UPDATE chapters SET updated_at=datetime('now') WHERE id=NEW.id;
END;

-- Cascade chapter writes to project.updated_at.
CREATE TRIGGER IF NOT EXISTS chapters_touch_project_ai
AFTER INSERT ON chapters
BEGIN
    UPDATE projects SET updated_at=datetime('now') WHERE id=NEW.project_id;
END;
CREATE TRIGGER IF NOT EXISTS chapters_touch_project_au
AFTER UPDATE ON chapters
BEGIN
    UPDATE projects SET updated_at=datetime('now') WHERE id=NEW.project_id;
END;
CREATE TRIGGER IF NOT EXISTS chapters_touch_project_ad
AFTER DELETE ON chapters
BEGIN
    UPDATE projects SET updated_at=datetime('now') WHERE id=OLD.project_id;
END;

-- Cascade task lifecycle to chapter.updated_at — claim/complete/fail/enqueue
-- all bump the chapter so the UI can show "đang chạy" freshness without a
-- separate column.
CREATE TRIGGER IF NOT EXISTS tasks_touch_chapter_ai
AFTER INSERT ON tasks
BEGIN
    UPDATE chapters SET updated_at=datetime('now') WHERE id=NEW.chapter_id;
END;
CREATE TRIGGER IF NOT EXISTS tasks_touch_chapter_au
AFTER UPDATE ON tasks
BEGIN
    UPDATE chapters SET updated_at=datetime('now') WHERE id=NEW.chapter_id;
END;
CREATE TRIGGER IF NOT EXISTS tasks_touch_chapter_ad
AFTER DELETE ON tasks
BEGIN
    UPDATE chapters SET updated_at=datetime('now') WHERE id=OLD.chapter_id;
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
        await _migrate_projects(db)
        await _migrate_chapters(db)
        # Triggers go last so the migration backfill above does not cascade.
        await db.executescript(_TRIGGERS)
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
            "INSERT INTO projects (slug, title, source_lang, target_lang, source_url, updated_at) "
            "VALUES (?,?,?,?,?, datetime('now'))",
            (slug, title, source_lang, target_lang, source_url),
        )
        await self._db.commit()
        return cur.lastrowid  # type: ignore

    async def get_project(self, project_id: int) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_project_by_slug(self, slug: str) -> dict | None:
        cur = await self._db.execute("SELECT * FROM projects WHERE slug=?", (slug,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def list_projects(self) -> list[dict]:
        cur = await self._db.execute("SELECT * FROM projects ORDER BY id DESC")
        return [dict(r) for r in await cur.fetchall()]

    async def update_project_metadata(
        self,
        project_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        cover_path: str | None = None,
        target_lang: str | None = None,
        settings: dict | None = None,
    ) -> None:
        sets: list[str] = []
        args: list = []
        if title is not None:
            sets.append("title=?"); args.append(title)
        if description is not None:
            sets.append("description=?"); args.append(description)
        if cover_path is not None:
            sets.append("cover_path=?"); args.append(cover_path)
        if target_lang is not None:
            sets.append("target_lang=?"); args.append(target_lang)
        if settings is not None:
            sets.append("settings=?"); args.append(json.dumps(settings, ensure_ascii=False))
        if not sets:
            return
        args.append(project_id)
        await self._db.execute(
            f"UPDATE projects SET {', '.join(sets)} WHERE id=?",
            tuple(args),
        )
        await self._db.commit()

    async def set_project_source_url(self, project_id: int, url: str) -> None:
        """One-shot helper for backfilling source_url on the first pull."""
        await self._db.execute(
            "UPDATE projects SET source_url=? WHERE id=?",
            (url, project_id),
        )
        await self._db.commit()

    async def get_project_settings(self, project_id: int) -> dict:
        cur = await self._db.execute(
            "SELECT settings FROM projects WHERE id=?", (project_id,)
        )
        row = await cur.fetchone()
        if not row or not row["settings"]:
            return {}
        try:
            return json.loads(row["settings"])
        except (TypeError, ValueError):
            return {}

    # ── Chapters ──────────────────────────────────────────────────

    async def get_or_create_chapter(
        self,
        project_id: int,
        idx: float,
        source_url: str | None = None,
        title: str | None = None,
    ) -> int:
        """Return chapter_id (existing or new). Updates title if provided."""
        cur = await self._db.execute(
            "SELECT id FROM chapters WHERE project_id=? AND idx=?", (project_id, idx)
        )
        row = await cur.fetchone()
        if row:
            if title is not None:
                await self._db.execute(
                    "UPDATE chapters SET title=? WHERE id=? AND (title IS NULL OR title='')",
                    (title, row["id"]),
                )
                await self._db.commit()
            return row["id"]
        cur = await self._db.execute(
            "INSERT INTO chapters (project_id, idx, source_url, title) VALUES (?,?,?,?)",
            (project_id, idx, source_url, title),
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

    async def get_chapters_with_status(self, project_id: int) -> list[dict]:
        """Chapters with state/stage/page_count, derived from DB only."""
        chapters = await self.get_all_chapters(project_id)
        result = []
        for ch in chapters:
            tasks = await self.get_tasks(ch["id"])
            state, stage, error = _derive_state(ch, tasks)
            result.append({
                "chapter_id": ch["id"],
                "project_id": ch["project_id"],
                "idx":        ch["idx"],
                "title":      ch.get("title"),
                "state":      state,
                "stage":      stage,
                "page_count": int(ch.get("page_count") or 0),
                "error":      error,
                "updated_at": ch.get("updated_at") or ch.get("created_at"),
            })
        return result

    async def get_chapter_with_status(self, chapter_id: int, project_id: int) -> dict | None:
        """Single chapter with state/stage/page_count/progress."""
        ch = await self.get_chapter(chapter_id)
        if ch is None or ch["project_id"] != project_id:
            return None
        tasks    = await self.get_tasks(chapter_id)
        state, stage, error = _derive_state(ch, tasks)
        progress = await self.get_chapter_progress(chapter_id)
        return {
            "chapter_id": chapter_id,
            "project_id": project_id,
            "idx":        ch["idx"],
            "title":      ch.get("title"),
            "state":      state,
            "stage":      stage,
            "page_count": int(ch.get("page_count") or 0),
            "error":      error,
            "updated_at": ch.get("updated_at") or ch.get("created_at"),
            "progress":   progress and {
                "stage":      progress["stage"],
                "page_index": progress["page_index"],
                "page_total": progress["page_total"],
            },
        }

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
        """Delete all derived data for a chapter — keeps identity + page_count.

        Resets `rendered=0` so a stale flag does not survive a reprepare.
        The prepared archive on the artifact store is kept intact; scan
        re-derives geometry/masks from prepared pixels.
        """
        try:
            await self._db.execute("BEGIN")
            for table in (
                "bubbles", "translations", "chapter_briefs", "tasks",
                "bubble_geometry", "page_geometry",
            ):
                await self._db.execute(
                    f"DELETE FROM {table} WHERE chapter_id=?", (chapter_id,),
                )
            await self._db.execute(
                "UPDATE chapters SET rendered=0 WHERE id=?",
                (chapter_id,),
            )
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    async def has_bubbles(self, chapter_id: int) -> bool:
        cur = await self._db.execute(
            "SELECT 1 FROM bubbles WHERE chapter_id=? LIMIT 1", (chapter_id,)
        )
        return await cur.fetchone() is not None

    # ── Geometry ──────────────────────────────────────────────────

    async def save_geometry(
        self, chapter_id: int, pages: list[dict],
    ) -> None:
        """Replace all geometry rows for a chapter atomically.

        `pages`: list of {page_index, width, height,
                          bubbles: [{bubble_idx, polygon, fit_box, erase_box, text_box}]}
        Each polygon/box value is a list (serialized to JSON for storage).
        """
        bubble_rows = [
            (
                chapter_id, p["page_index"], b["bubble_idx"],
                json.dumps(b["polygon"]),
                json.dumps(b["fit_box"]),
                json.dumps(b["erase_box"]),
                json.dumps(b["text_box"]),
            )
            for p in pages for b in p["bubbles"]
        ]
        try:
            await self._db.execute("BEGIN")
            await self._db.execute(
                "DELETE FROM page_geometry WHERE chapter_id=?", (chapter_id,),
            )
            await self._db.execute(
                "DELETE FROM bubble_geometry WHERE chapter_id=?", (chapter_id,),
            )
            await self._db.executemany(
                "INSERT INTO page_geometry (chapter_id, page_index, width, height) "
                "VALUES (?,?,?,?)",
                [(chapter_id, p["page_index"], p["width"], p["height"]) for p in pages],
            )
            if bubble_rows:
                await self._db.executemany(
                    "INSERT INTO bubble_geometry "
                    "(chapter_id, page_index, bubble_idx, polygon, fit_box, erase_box, text_box) "
                    "VALUES (?,?,?,?,?,?,?)",
                    bubble_rows,
                )
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise

    async def get_geometry(self, chapter_id: int) -> list[dict]:
        """Return chapter geometry as list of page dicts.

        Same shape as save_geometry input (lists, not JSON strings).
        """
        cur = await self._db.execute(
            "SELECT page_index, width, height FROM page_geometry "
            "WHERE chapter_id=? ORDER BY page_index",
            (chapter_id,),
        )
        page_rows = [dict(r) for r in await cur.fetchall()]
        cur = await self._db.execute(
            "SELECT page_index, bubble_idx, polygon, fit_box, erase_box, text_box "
            "FROM bubble_geometry WHERE chapter_id=? ORDER BY page_index, bubble_idx",
            (chapter_id,),
        )
        page_bubbles: dict[int, list[dict]] = {}
        for r in await cur.fetchall():
            page_bubbles.setdefault(r["page_index"], []).append({
                "bubble_idx": r["bubble_idx"],
                "polygon":   json.loads(r["polygon"]),
                "fit_box":   json.loads(r["fit_box"]),
                "erase_box": json.loads(r["erase_box"]),
                "text_box":  json.loads(r["text_box"]),
            })
        return [
            {
                "page_index": p["page_index"],
                "width":      p["width"],
                "height":     p["height"],
                "bubbles":    page_bubbles.get(p["page_index"], []),
            }
            for p in page_rows
        ]

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

    async def update_translation(
        self,
        chapter_id: int,
        page_index: int,
        bubble_idx: int,
        translated_text: str,
        kind: str | None = None,
    ) -> bool:
        """Patch one translation row. Returns True if a row was updated.

        Used by the manual-edit endpoint. Caller is responsible for
        re-enqueueing render so the change appears on the rendered page.
        """
        sets = ["translated_text=?"]
        args: list = [translated_text]
        if kind is not None:
            sets.append("kind=?"); args.append(kind)
        args += [chapter_id, page_index, bubble_idx]
        cur = await self._db.execute(
            f"UPDATE translations SET {', '.join(sets)} "
            "WHERE chapter_id=? AND page_index=? AND bubble_idx=?",
            tuple(args),
        )
        await self._db.commit()
        return cur.rowcount > 0

    # ── Glossary ──────────────────────────────────────────────────

    async def get_glossary(self, project_id: int) -> dict[str, str]:
        cur = await self._db.execute(
            "SELECT source_term, target_term FROM glossary WHERE project_id=?", (project_id,)
        )
        return {r["source_term"]: r["target_term"] for r in await cur.fetchall()}

    async def list_glossary(self, project_id: int) -> list[dict]:
        """Full rows ordered by source_term — used by the API/UI."""
        cur = await self._db.execute(
            "SELECT id, source_term, target_term, notes "
            "FROM glossary WHERE project_id=? ORDER BY source_term",
            (project_id,),
        )
        return [dict(r) for r in await cur.fetchall()]

    async def upsert_glossary_term(
        self,
        project_id: int,
        source_term: str,
        target_term: str,
        notes: str | None = None,
    ) -> int:
        await self._db.execute(
            "INSERT INTO glossary (project_id, source_term, target_term, notes) "
            "VALUES (?,?,?,?) "
            "ON CONFLICT(project_id, source_term) DO UPDATE SET "
            "  target_term=excluded.target_term, notes=excluded.notes",
            (project_id, source_term, target_term, notes),
        )
        await self._db.commit()
        cur = await self._db.execute(
            "SELECT id FROM glossary WHERE project_id=? AND source_term=?",
            (project_id, source_term),
        )
        row = await cur.fetchone()
        return row["id"] if row else 0

    async def delete_glossary_term(self, project_id: int, term_id: int) -> bool:
        cur = await self._db.execute(
            "DELETE FROM glossary WHERE id=? AND project_id=?",
            (term_id, project_id),
        )
        await self._db.commit()
        return cur.rowcount > 0

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

    async def delete_project(self, project_id: int) -> None:
        await self._db.execute("DELETE FROM projects WHERE id=?", (project_id,))
        await self._db.commit()

    async def delete_chapter(self, chapter_id: int) -> bool:
        """Drop the chapter row entirely (cascades to bubbles/translations/
        tasks/geometry/briefs via FK ON DELETE CASCADE).

        Caller is responsible for removing the chapter's artifacts
        (prepared.bnl, render.bnl, masks.npz) from the artifact store.
        """
        cur = await self._db.execute("DELETE FROM chapters WHERE id=?", (chapter_id,))
        await self._db.commit()
        return cur.rowcount > 0

    # ── Queue stats (Tier B — workers dashboard) ──────────────────

    async def queue_stats(self) -> dict:
        """Aggregated task counts and recent claim activity per stage.

        Shape:
            {
              "stages": {
                "scan":      {"pending": N, "running": M, "stale": K},
                "translate": {...},
                "render":    {...},
              },
              "active_workers": ["uuid-1", "uuid-2", ...],
            }
        A task is "running" if claimed_by IS NOT NULL and claimed within
        the last 10 minutes; "stale" if claim is older than that (eligible
        for re-claim by another worker).
        """
        cur = await self._db.execute(
            "SELECT stage, "
            "  SUM(CASE WHEN claimed_by IS NULL THEN 1 ELSE 0 END) AS pending, "
            "  SUM(CASE WHEN claimed_by IS NOT NULL "
            "       AND claimed_at >= datetime('now','-10 minutes') THEN 1 ELSE 0 END) AS running, "
            "  SUM(CASE WHEN claimed_by IS NOT NULL "
            "       AND claimed_at <  datetime('now','-10 minutes') THEN 1 ELSE 0 END) AS stale "
            "FROM tasks GROUP BY stage"
        )
        stages: dict[str, dict[str, int]] = {}
        for row in await cur.fetchall():
            stages[row["stage"]] = {
                "pending": int(row["pending"] or 0),
                "running": int(row["running"] or 0),
                "stale":   int(row["stale"] or 0),
            }
        cur = await self._db.execute(
            "SELECT DISTINCT claimed_by FROM tasks "
            "WHERE claimed_by IS NOT NULL "
            "  AND claimed_at >= datetime('now','-10 minutes')"
        )
        active = [r["claimed_by"] for r in await cur.fetchall()]
        return {"stages": stages, "active_workers": active}

    # ── Users / identities ────────────────────────────────────────

    async def upsert_user_from_identity(
        self,
        *,
        provider:    str,
        external_id: str,
        display_name: str,
        avatar_url:  str | None = None,
        email:       str | None = None,
        metadata:    dict | None = None,
        promote_admin: bool = False,
    ) -> dict:
        """Find-or-create user from a (provider, external_id) tuple.

        - If identity exists → update last_login_at + display_name/avatar/email,
          return the linked user.
        - If identity missing → create new user + identity in one transaction.
        - `promote_admin=True` sets tier='admin' on user creation, used by
          the bootstrap admin flow.
        """
        cur = await self._db.execute(
            "SELECT user_id FROM identities WHERE provider=? AND external_id=?",
            (provider, external_id),
        )
        row = await cur.fetchone()

        if row:
            user_id = row["user_id"]
            await self._db.execute(
                "UPDATE users SET display_name=?, avatar_url=?, email=COALESCE(?, email), "
                "  last_login_at=datetime('now') WHERE id=?",
                (display_name, avatar_url, email, user_id),
            )
            if metadata is not None:
                await self._db.execute(
                    "UPDATE identities SET metadata=? "
                    "WHERE provider=? AND external_id=?",
                    (json.dumps(metadata, ensure_ascii=False),
                     provider, external_id),
                )
            await self._db.commit()
            return await self.get_user(user_id)  # type: ignore[return-value]

        # New user
        try:
            await self._db.execute("BEGIN")
            cur = await self._db.execute(
                "INSERT INTO users (display_name, avatar_url, email, tier, last_login_at) "
                "VALUES (?,?,?,?, datetime('now'))",
                (display_name, avatar_url, email,
                 "admin" if promote_admin else "member"),
            )
            user_id = cur.lastrowid
            await self._db.execute(
                "INSERT INTO identities (user_id, provider, external_id, metadata) "
                "VALUES (?,?,?,?)",
                (user_id, provider, external_id,
                 json.dumps(metadata or {}, ensure_ascii=False)),
            )
            await self._db.commit()
        except Exception:
            await self._db.rollback()
            raise
        return await self.get_user(user_id)  # type: ignore[return-value]

    async def get_user(self, user_id: int) -> dict | None:
        cur = await self._db.execute("SELECT * FROM users WHERE id=?", (user_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def get_user_by_identity(
        self, provider: str, external_id: str,
    ) -> dict | None:
        cur = await self._db.execute(
            "SELECT u.* FROM users u "
            "JOIN identities i ON i.user_id=u.id "
            "WHERE i.provider=? AND i.external_id=?",
            (provider, external_id),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    async def set_user_tier(self, user_id: int, tier: str) -> None:
        await self._db.execute(
            "UPDATE users SET tier=? WHERE id=?", (tier, user_id),
        )
        await self._db.commit()

    # ── Events ────────────────────────────────────────────────────

    async def append_event(self, data: dict) -> None:
        import json
        await self._db.execute("INSERT INTO events (data) VALUES (?)", (json.dumps(data),))
        await self._db.commit()

    async def get_events_after(self, seq: int) -> list[dict]:
        import json
        cur = await self._db.execute(
            "SELECT id, data FROM events WHERE id > ? ORDER BY id LIMIT 100", (seq,)
        )
        return [{"id": r["id"], **json.loads(r["data"])} async for r in cur]

    async def get_chapter_progress(self, chapter_id: int) -> dict | None:
        """Latest PageDone event for this chapter — for page-level progress."""
        import json
        cur = await self._db.execute(
            "SELECT data FROM events WHERE json_extract(data,'$.type')='PageDone' "
            "AND json_extract(data,'$.chapter_id')=? ORDER BY id DESC LIMIT 1",
            (chapter_id,),
        )
        row = await cur.fetchone()
        return json.loads(row["data"]) if row else None

    # ── Chapter archive state ─────────────────────────────────────

    async def set_prepared_done(self, chapter_id: int, page_count: int) -> None:
        """Mark chapter as prepared. Resets rendered=0 so a stale render does
        not survive a reprepare."""
        await self._db.execute(
            "UPDATE chapters SET page_count=?, rendered=0 WHERE id=?",
            (page_count, chapter_id),
        )
        await self._db.commit()

    async def set_rendered(self, chapter_id: int, rendered: bool) -> None:
        """Flip the persistent `rendered` flag. The tasks table is the only
        source of truth for render-in-flight; a single render worker is
        enforced via `claim_task`."""
        await self._db.execute(
            "UPDATE chapters SET rendered=? WHERE id=?",
            (1 if rendered else 0, chapter_id),
        )
        await self._db.commit()

    async def get_chapter_render_state(self, chapter_id: int) -> dict | None:
        cur = await self._db.execute(
            "SELECT rendered, page_count FROM chapters WHERE id=?",
            (chapter_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {"rendered": bool(row["rendered"]), "page_count": row["page_count"]}


# ── Helpers ───────────────────────────────────────────────────────────

_FTS_SPECIAL = _re.compile(r"[\"'\(\)\*\:\^\-\+\{\}\[\]~]")


def _derive_state(chapter_row: dict, tasks: list[dict]) -> tuple[str, str, str]:
    """Derive (state, stage, error) from chapter row + tasks table.

    Priority:
      running task         → running    (re-scan/re-render in progress)
      task with attempts≥3 → error
      pending task         → pending
      chapters.rendered=1  → done
      otherwise            → idle
    """
    running = [t for t in tasks if t["claimed_by"]]
    if running:
        return "running", running[0]["stage"], ""
    failed = [t for t in tasks if t["last_error"] and t["attempts"] >= 3]
    if failed:
        return "error", failed[0]["stage"], failed[0]["last_error"] or ""
    pending = [t for t in tasks if not t["claimed_by"]]
    if pending:
        return "pending", pending[0]["stage"], ""
    if chapter_row.get("rendered"):
        return "done", "", ""
    return "idle", "", ""

def _fts_escape(query: str) -> str:
    clean = _FTS_SPECIAL.sub(" ", query).strip()
    words = clean.split()
    return " ".join(f'"{w}"' for w in words) if words else ""
