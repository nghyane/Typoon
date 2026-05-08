"""Postgres storage — identity, knowledge, worker coordination.

Schema lives in `schema.sql` and is applied idempotently at `open()`;
there is no migration tooling — bump SCHEMA_VERSION + drop/recreate the
database during dev when the shape changes.

Datetime handling: asyncpg returns `datetime` objects for TIMESTAMPTZ.
We convert each timestamp to RFC 3339 in UTC at the SQL layer
(`to_char(... AT TIME ZONE 'UTC', ... )`) so callers see strings only.
"""

from __future__ import annotations

import json
import logging
import re as _re
from datetime import datetime, timezone

import asyncpg

logger = logging.getLogger(__name__)

# Bump this when schema.sql changes shape. Mismatch on boot ⇒ refuse to
# start, instruct the operator to nuke the volume.
SCHEMA_VERSION = "6"

# Hard cap on retry attempts per task. Deterministic crashes (NameError,
# malformed input, persistent OOM) must not loop forever — the worker
# would otherwise burn CPU on a dead chapter and starve the queue. After
# this many failures the task is dead-lettered: visible to status views
# (last_error populated) but never re-claimed until an operator redoes it.
MAX_TASK_ATTEMPTS = 3

# How long a claim is considered "live". After this, the task is
# re-claimable by another worker (see claim_task) AND status views
# treat it as pending again rather than running. Without this, a worker
# that crashes hard (OOM, killed) would leave its chapter forever
# stuck on "running" in the UI even though no one is working on it.
STALE_CLAIM_SECONDS = 10 * 60


def _read_schema_sql() -> str:
    from pathlib import Path
    return (Path(__file__).parent / "schema.sql").read_text()


# ── Postgres FTS query sanitiser ──────────────────────────────────────
#
# The agent passes natural strings (`"phép thuật"`, `magic OR sorcery`,
# `-cấm`) through search_knowledge. `websearch_to_tsquery` accepts
# Google-style syntax directly — no escaping needed for the common case.
# We only strip control chars that asyncpg refuses.
_CTRL = _re.compile(r"[\x00-\x1f]")


def _clean_query(q: str) -> str:
    return _CTRL.sub(" ", q).strip()


# ── ISO-string timestamp formatting ───────────────────────────────────
# Convert all timestamps to UTC and format as RFC 3339 with a `Z`
# suffix — directly parseable by JS `new Date(...)` and avoids local
# time zone surprises across hosts.
_ISO_FMT = "YYYY-MM-DD\"T\"HH24:MI:SS\"Z\""


def _ts(col: str) -> str:
    return f"to_char(({col}) AT TIME ZONE 'UTC', '{_ISO_FMT}') AS {col}"


_TS_PROJECTS = (
    "id, slug, title, description, cover_path, source_url, "
    "source_lang, target_lang, owner_id, shared, settings, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_CHAPTERS = (
    "id, project_id, idx, title, source_url, rendered, page_count, "
    "archive_backend, archive_locator, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_USERS = (
    "id, display_name, avatar_url, email, "
    f"{_ts('created_at')}, {_ts('last_login_at')}"
)


def _row_dict(row: asyncpg.Record | None) -> dict | None:
    return dict(row) if row else None


class PostgresStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @staticmethod
    async def open(dsn: str) -> "PostgresStore":
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"DATABASE_URL must be a postgresql:// DSN, got: {dsn!r}"
            )
        # statement_cache_size=0 disables asyncpg's per-connection prepared
        # statement cache. We've seen sporadic "_get_statement" failures
        # under concurrent first requests; the cache was a small win not
        # worth the flake. Re-enable later if profiling justifies it.
        pool = await asyncpg.create_pool(
            dsn, min_size=2, max_size=10, statement_cache_size=0,
        )
        async with pool.acquire() as conn:
            await conn.execute(_read_schema_sql())
            await _verify_schema_version(conn)
        return PostgresStore(pool)

    async def close(self) -> None:
        await self._pool.close()

    # ── Projects ──────────────────────────────────────────────────

    async def get_or_create_project(
        self,
        slug: str,
        title: str,
        source_lang: str,
        target_lang: str,
        source_url: str | None = None,
        owner_id: int | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT id FROM projects WHERE slug=$1", slug)
            if row:
                return row["id"]
            if source_url:
                row = await conn.fetchrow(
                    "SELECT id FROM projects WHERE source_url=$1", source_url,
                )
                if row:
                    return row["id"]
            row = await conn.fetchrow(
                "INSERT INTO projects (slug, title, source_lang, target_lang, source_url, owner_id) "
                "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                slug, title, source_lang, target_lang, source_url, owner_id,
            )
            return row["id"]

    async def get_project(self, project_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_PROJECTS} FROM projects WHERE id=$1", project_id,
            ))

    async def get_project_by_slug(self, slug: str) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_PROJECTS} FROM projects WHERE slug=$1", slug,
            ))

    async def list_projects(
        self,
        *,
        viewer_id: int | None = None,
        filter: str = "all",
    ) -> list[dict]:
        """List projects visible to `viewer_id`.

        filter:
          all       — owned by viewer + shared (default)
          mine      — owned by viewer
          pinned    — pinned by viewer
          community — shared && owner_id != viewer (others' shared)

        viewer_id=None returns everything (admin / migration use).
        Each row carries `is_pinned` (bool, false when no viewer).
        """
        if viewer_id is None:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"SELECT {_TS_PROJECTS}, FALSE AS is_pinned "
                    "FROM projects ORDER BY id DESC",
                )
            return [dict(r) for r in rows]

        select = (
            f"SELECT {_TS_PROJECTS}, "
            "  EXISTS(SELECT 1 FROM project_pins pp "
            "         WHERE pp.user_id=$1 AND pp.project_id=projects.id) "
            "  AS is_pinned "
            "FROM projects "
        )
        if filter == "mine":
            where = "WHERE owner_id = $1"
            params: list = [viewer_id]
        elif filter == "pinned":
            where = (
                "WHERE id IN (SELECT project_id FROM project_pins WHERE user_id=$1) "
                "  AND (owner_id = $1 OR shared = TRUE)"
            )
            params = [viewer_id]
        elif filter == "community":
            where = "WHERE shared = TRUE AND (owner_id IS NULL OR owner_id <> $1)"
            params = [viewer_id]
        else:  # all
            where = "WHERE owner_id = $1 OR shared = TRUE"
            params = [viewer_id]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                select + where + " ORDER BY updated_at DESC, id DESC",
                *params,
            )
        return [dict(r) for r in rows]

    async def can_view_project(self, project_id: int, viewer_id: int) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM projects "
                "WHERE id=$1 AND (owner_id=$2 OR shared=TRUE) LIMIT 1",
                project_id, viewer_id,
            )
        return row is not None

    async def is_project_owner(self, project_id: int, user_id: int) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM projects WHERE id=$1 AND owner_id=$2 LIMIT 1",
                project_id, user_id,
            )
        return row is not None

    # ── Pins ──────────────────────────────────────────────────────

    async def pin_project(self, user_id: int, project_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO project_pins (user_id, project_id) VALUES ($1,$2) "
                "ON CONFLICT DO NOTHING",
                user_id, project_id,
            )

    async def unpin_project(self, user_id: int, project_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM project_pins WHERE user_id=$1 AND project_id=$2",
                user_id, project_id,
            )

    async def is_pinned(self, user_id: int, project_id: int) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM project_pins WHERE user_id=$1 AND project_id=$2",
                user_id, project_id,
            )
        return row is not None

    async def update_project_metadata(
        self,
        project_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        cover_path: str | None = None,
        target_lang: str | None = None,
        settings: dict | None = None,
        shared: bool | None = None,
        owner_id: int | None = None,
    ) -> None:
        sets: list[str] = []
        args: list = []
        if title is not None:
            args.append(title); sets.append(f"title=${len(args)}")
        if description is not None:
            args.append(description); sets.append(f"description=${len(args)}")
        if cover_path is not None:
            args.append(cover_path); sets.append(f"cover_path=${len(args)}")
        if target_lang is not None:
            args.append(target_lang); sets.append(f"target_lang=${len(args)}")
        if settings is not None:
            args.append(json.dumps(settings, ensure_ascii=False))
            sets.append(f"settings=${len(args)}::jsonb")
        if shared is not None:
            args.append(shared); sets.append(f"shared=${len(args)}")
        if owner_id is not None:
            args.append(owner_id); sets.append(f"owner_id=${len(args)}")
        if not sets:
            return
        args.append(project_id)
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE projects SET {', '.join(sets)} WHERE id=${len(args)}",
                *args,
            )

    async def set_project_source_url(self, project_id: int, url: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE projects SET source_url=$1 WHERE id=$2", url, project_id,
            )

    async def get_project_settings(self, project_id: int) -> dict:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT settings FROM projects WHERE id=$1", project_id,
            )
        if not row or not row["settings"]:
            return {}
        return _to_jsonable_dict(row["settings"])

    # ── Chapters ──────────────────────────────────────────────────

    async def get_or_create_chapter(
        self,
        project_id: int,
        idx: float,
        source_url: str | None = None,
        title: str | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, title FROM chapters WHERE project_id=$1 AND idx=$2",
                project_id, idx,
            )
            if row:
                if title is not None and not row["title"]:
                    await conn.execute(
                        "UPDATE chapters SET title=$1 WHERE id=$2 AND (title IS NULL OR title='')",
                        title, row["id"],
                    )
                return row["id"]
            row = await conn.fetchrow(
                "INSERT INTO chapters (project_id, idx, source_url, title) "
                "VALUES ($1,$2,$3,$4) RETURNING id",
                project_id, idx, source_url, title,
            )
            return row["id"]

    async def get_chapter(self, chapter_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_CHAPTERS} FROM chapters WHERE id=$1", chapter_id,
            ))

    async def get_chapter_by_idx(self, project_id: int, idx: float) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_CHAPTERS} FROM chapters WHERE project_id=$1 AND idx=$2",
                project_id, idx,
            ))

    async def get_all_chapters(self, project_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_TS_CHAPTERS} FROM chapters WHERE project_id=$1 ORDER BY idx",
                project_id,
            )
            return [dict(r) for r in rows]

    async def get_chapters_with_status(self, project_id: int) -> list[dict]:
        chapters = await self.get_all_chapters(project_id)
        result = []
        for ch in chapters:
            tasks = await self.get_tasks(ch["id"])
            state, stage, error = _derive_state(ch, tasks)
            result.append({
                "chapter_id":      ch["id"],
                "project_id":      ch["project_id"],
                "idx":             ch["idx"],
                "title":           ch.get("title"),
                "state":           state,
                "stage":           stage,
                "page_count":      int(ch.get("page_count") or 0),
                "error":           error,
                "updated_at":      ch.get("updated_at") or ch.get("created_at"),
                "archive_backend": ch.get("archive_backend"),
                "archive_locator": ch.get("archive_locator"),
            })
        return result

    async def get_chapter_with_status(
        self, chapter_id: int, project_id: int,
    ) -> dict | None:
        ch = await self.get_chapter(chapter_id)
        if ch is None or ch["project_id"] != project_id:
            return None
        tasks    = await self.get_tasks(chapter_id)
        state, stage, error = _derive_state(ch, tasks)
        progress = await self.get_chapter_progress(chapter_id)
        return {
            "chapter_id":      chapter_id,
            "project_id":      project_id,
            "idx":             ch["idx"],
            "title":           ch.get("title"),
            "state":           state,
            "stage":           stage,
            "page_count":      int(ch.get("page_count") or 0),
            "error":           error,
            "updated_at":      ch.get("updated_at") or ch.get("created_at"),
            "archive_backend": ch.get("archive_backend"),
            "archive_locator": ch.get("archive_locator"),
            "progress":        progress and {
                "stage":      progress["stage"],
                "page_index": progress["page_index"],
                "page_total": progress["page_total"],
            },
        }

    # ── Tasks ─────────────────────────────────────────────────────

    async def enqueue(self, chapter_id: int, stage: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tasks (chapter_id, stage) VALUES ($1,$2) "
                "ON CONFLICT (chapter_id, stage) DO NOTHING",
                chapter_id, stage,
            )

    async def enqueue_many(self, chapter_id: int, stages: list[str]) -> None:
        if not stages:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO tasks (chapter_id, stage) VALUES ($1,$2) "
                "ON CONFLICT (chapter_id, stage) DO NOTHING",
                [(chapter_id, s) for s in stages],
            )

    async def claim_task(self, stage: str, worker_id: str) -> int | None:
        """Atomically claim one pending task. Returns chapter_id or None.

        FOR UPDATE SKIP LOCKED gives us a single-statement claim with no
        race recheck needed — the row we get back is definitively ours.
        Tasks that exceeded MAX_TASK_ATTEMPTS are dead-lettered and never
        re-claimed; an operator must redo or delete them.
        Stale claims (worker crashed without releasing) older than
        STALE_CLAIM_SECONDS become re-claimable.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                UPDATE tasks
                SET    claimed_by = $1, claimed_at = NOW()
                WHERE  ctid = (
                    SELECT ctid FROM tasks
                    WHERE  stage = $2
                      AND  attempts < $3
                      AND  (claimed_by IS NULL
                            OR claimed_at < NOW() - INTERVAL '{STALE_CLAIM_SECONDS} seconds')
                    ORDER  BY chapter_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT  1
                )
                RETURNING chapter_id
                """,
                worker_id, stage, MAX_TASK_ATTEMPTS,
            )
        return row["chapter_id"] if row else None

    async def complete_task(self, chapter_id: int, stage: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM tasks WHERE chapter_id=$1 AND stage=$2",
                chapter_id, stage,
            )

    async def advance_task(
        self, chapter_id: int, completed_stage: str, next_stage: str,
    ) -> None:
        """Atomically complete a stage and enqueue the next one.

        Without this, a chapter briefly has zero tasks between
        complete_task and enqueue, which surfaces as an "idle" flicker
        in the UI on every stage handoff. One transaction → no gap.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM tasks WHERE chapter_id=$1 AND stage=$2",
                chapter_id, completed_stage,
            )
            await conn.execute(
                "INSERT INTO tasks (chapter_id, stage) VALUES ($1,$2) "
                "ON CONFLICT (chapter_id, stage) DO NOTHING",
                chapter_id, next_stage,
            )

    async def fail_task(self, chapter_id: int, stage: str, error: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE tasks
                SET    claimed_by = NULL,
                       claimed_at = NULL,
                       attempts   = attempts + 1,
                       last_error = $1
                WHERE  chapter_id=$2 AND stage=$3
                """,
                error, chapter_id, stage,
            )

    async def get_tasks(self, chapter_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT chapter_id, stage, claimed_by, "
                f"{_ts('claimed_at')}, "
                "attempts, last_error "
                "FROM tasks WHERE chapter_id=$1",
                chapter_id,
            )
            return [dict(r) for r in rows]

    async def delete_tasks_from(self, chapter_id: int, stage: str) -> None:
        order = ["prepare", "scan", "translate", "render"]
        if stage not in order:
            return
        stages_to_delete = order[order.index(stage):]
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM tasks WHERE chapter_id=$1 AND stage = ANY($2::text[])",
                chapter_id, stages_to_delete,
            )

    # ── Bubbles ───────────────────────────────────────────────────

    async def save_bubbles(self, chapter_id: int, bubbles: list[dict]) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM bubbles WHERE chapter_id=$1", chapter_id)
            await conn.executemany(
                "INSERT INTO bubbles (chapter_id, page_index, bubble_idx, "
                "source_text, confidence, shape_kind) "
                "VALUES ($1,$2,$3,$4,$5,$6)",
                [
                    (chapter_id, b["page_index"], b["bubble_idx"],
                     b["source_text"], b["confidence"],
                     b.get("shape_kind", "dialogue"))
                    for b in bubbles
                ],
            )

    async def get_bubbles(self, chapter_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT page_index, bubble_idx, source_text, confidence, shape_kind "
                "FROM bubbles WHERE chapter_id=$1 "
                "ORDER BY page_index, bubble_idx",
                chapter_id,
            )
            return [dict(r) for r in rows]

    async def has_bubbles(self, chapter_id: int) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM bubbles WHERE chapter_id=$1 LIMIT 1", chapter_id,
            )
        return row is not None

    async def delete_chapter_data(self, chapter_id: int) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            for table in (
                "bubbles", "translations", "chapter_briefs", "tasks",
                "bubble_geometry", "page_geometry",
            ):
                await conn.execute(
                    f"DELETE FROM {table} WHERE chapter_id=$1", chapter_id,
                )
            await conn.execute(
                "UPDATE chapters SET rendered=FALSE, "
                "archive_backend=NULL, archive_locator=NULL "
                "WHERE id=$1", chapter_id,
            )

    # ── Geometry ──────────────────────────────────────────────────

    async def save_geometry(self, chapter_id: int, pages: list[dict]) -> None:
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
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM page_geometry WHERE chapter_id=$1", chapter_id)
            await conn.execute("DELETE FROM bubble_geometry WHERE chapter_id=$1", chapter_id)
            await conn.executemany(
                "INSERT INTO page_geometry (chapter_id, page_index, width, height) "
                "VALUES ($1,$2,$3,$4)",
                [(chapter_id, p["page_index"], p["width"], p["height"]) for p in pages],
            )
            if bubble_rows:
                await conn.executemany(
                    "INSERT INTO bubble_geometry "
                    "(chapter_id, page_index, bubble_idx, polygon, fit_box, erase_box, text_box) "
                    "VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6::jsonb,$7::jsonb)",
                    bubble_rows,
                )

    async def get_geometry(self, chapter_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            page_rows = await conn.fetch(
                "SELECT page_index, width, height FROM page_geometry "
                "WHERE chapter_id=$1 ORDER BY page_index", chapter_id,
            )
            bubble_rows = await conn.fetch(
                "SELECT page_index, bubble_idx, polygon, fit_box, erase_box, text_box "
                "FROM bubble_geometry WHERE chapter_id=$1 "
                "ORDER BY page_index, bubble_idx", chapter_id,
            )
        page_bubbles: dict[int, list[dict]] = {}
        for r in bubble_rows:
            page_bubbles.setdefault(r["page_index"], []).append({
                "bubble_idx": r["bubble_idx"],
                "polygon":   _to_jsonable(r["polygon"]),
                "fit_box":   _to_jsonable(r["fit_box"]),
                "erase_box": _to_jsonable(r["erase_box"]),
                "text_box":  _to_jsonable(r["text_box"]),
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

    async def save_translations(self, chapter_id: int, records: list[dict]) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM translations WHERE chapter_id=$1", chapter_id,
            )
            await conn.executemany(
                "INSERT INTO translations "
                "(chapter_id, page_index, bubble_idx, translated_text, kind) "
                "VALUES ($1,$2,$3,$4,$5)",
                [
                    (chapter_id, r["page_index"], r["bubble_idx"],
                     r["translated_text"], r["kind"])
                    for r in records
                ],
            )

    async def get_translations(self, chapter_id: int) -> dict[tuple[int, int], dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT page_index, bubble_idx, translated_text, kind "
                "FROM translations WHERE chapter_id=$1 "
                "ORDER BY page_index, bubble_idx", chapter_id,
            )
        return {(r["page_index"], r["bubble_idx"]): dict(r) for r in rows}

    async def has_translations(self, chapter_id: int) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM translations WHERE chapter_id=$1 LIMIT 1", chapter_id,
            )
        return row is not None

    async def update_translation(
        self,
        chapter_id: int,
        page_index: int,
        bubble_idx: int,
        translated_text: str,
        kind: str | None = None,
    ) -> bool:
        sets = ["translated_text=$1"]
        args: list = [translated_text]
        if kind is not None:
            args.append(kind); sets.append(f"kind=${len(args)}")
        args.extend([chapter_id, page_index, bubble_idx])
        n = len(args)
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE translations SET {', '.join(sets)} "
                f"WHERE chapter_id=${n-2} AND page_index=${n-1} AND bubble_idx=${n}",
                *args,
            )
        # asyncpg returns "UPDATE N"
        return result.startswith("UPDATE ") and not result.endswith("0")

    # ── Glossary ──────────────────────────────────────────────────

    async def get_glossary(self, project_id: int) -> dict[str, str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source_term, target_term FROM glossary WHERE project_id=$1",
                project_id,
            )
        return {r["source_term"]: r["target_term"] for r in rows}

    async def list_glossary(self, project_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, source_term, target_term, notes "
                "FROM glossary WHERE project_id=$1 ORDER BY source_term",
                project_id,
            )
        return [dict(r) for r in rows]

    async def upsert_glossary_term(
        self,
        project_id: int,
        source_term: str,
        target_term: str,
        notes: str | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO glossary (project_id, source_term, target_term, notes) "
                "VALUES ($1,$2,$3,$4) "
                "ON CONFLICT(project_id, source_term) DO UPDATE SET "
                "  target_term=excluded.target_term, notes=excluded.notes "
                "RETURNING id",
                project_id, source_term, target_term, notes,
            )
        return row["id"] if row else 0

    async def delete_glossary_term(self, project_id: int, term_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM glossary WHERE id=$1 AND project_id=$2",
                term_id, project_id,
            )
        return result.startswith("DELETE ") and not result.endswith("0")

    async def glossary_search(self, project_id: int, query: str) -> list[dict]:
        clean = _clean_query(query)
        if not clean:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT source_term, target_term, notes "
                "FROM glossary "
                "WHERE project_id=$1 "
                "  AND source_term_tsv @@ websearch_to_tsquery('simple', $2) "
                "ORDER BY ts_rank(source_term_tsv, websearch_to_tsquery('simple', $2)) DESC "
                "LIMIT 10",
                project_id, clean,
            )
        return [dict(r) for r in rows]

    async def upsert_glossary_terms(self, project_id: int, terms: dict[str, str]) -> None:
        if not terms:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                "INSERT INTO glossary (project_id, source_term, target_term) "
                "VALUES ($1,$2,$3) "
                "ON CONFLICT(project_id, source_term) DO UPDATE SET "
                "  target_term=excluded.target_term",
                [(project_id, src, tgt) for src, tgt in terms.items()],
            )

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
        payload = (
            chapter_id,
            json.dumps(brief, ensure_ascii=False),
            str(brief.get("summary", "")),
            "\n".join(f"{k} -> {v}" for k, v in terms.items()),
            "\n".join(str(x) for x in brief.get("facts", []) or []),
            address_text + "\n" + "\n".join(str(x) for x in style),
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chapter_briefs "
                "(chapter_id, brief_json, summary, terms_text, facts_text, rules_text, updated_at) "
                "VALUES ($1, $2::jsonb, $3, $4, $5, $6, NOW()) "
                "ON CONFLICT (chapter_id) DO UPDATE SET "
                "  brief_json=excluded.brief_json, summary=excluded.summary, "
                "  terms_text=excluded.terms_text, facts_text=excluded.facts_text, "
                "  rules_text=excluded.rules_text, updated_at=NOW()",
                *payload,
            )
            if terms:
                row = await conn.fetchrow(
                    "SELECT project_id FROM chapters WHERE id=$1", chapter_id,
                )
                if row:
                    await conn.executemany(
                        "INSERT INTO glossary (project_id, source_term, target_term) "
                        "VALUES ($1,$2,$3) "
                        "ON CONFLICT(project_id, source_term) DO UPDATE SET "
                        "  target_term=excluded.target_term",
                        [(row["project_id"], src, tgt) for src, tgt in terms.items()],
                    )

    async def get_chapter_brief(self, chapter_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT brief_json, summary, terms_text, facts_text, rules_text "
                "FROM chapter_briefs WHERE chapter_id=$1", chapter_id,
            )
        if not row:
            return None
        out = dict(row)
        out["brief"] = _to_jsonable_dict(out.pop("brief_json"))
        return out

    async def get_recent_chapter_briefs(
        self,
        project_id: int,
        before_chapter_idx: float,
        limit: int = 3,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT c.idx, cb.brief_json, cb.summary, cb.terms_text, "
                "       cb.facts_text, cb.rules_text "
                "FROM chapter_briefs cb "
                "JOIN chapters c ON c.id = cb.chapter_id "
                "WHERE c.project_id=$1 AND c.idx<$2 "
                "ORDER BY c.idx DESC LIMIT $3",
                project_id, before_chapter_idx, limit,
            )
        result = []
        for row in rows:
            out = dict(row)
            out["chapter"] = out.pop("idx")
            out["brief"] = _to_jsonable_dict(out.pop("brief_json"))
            result.append(out)
        return result

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
        async with self._pool.acquire() as conn:
            for query in queries:
                clean = _clean_query(query)
                if not clean:
                    continue
                if before_chapter_idx is not None:
                    rows = await conn.fetch(
                        "SELECT c.idx, cb.summary, cb.terms_text, cb.facts_text, cb.rules_text "
                        "FROM chapter_briefs cb "
                        "JOIN chapters c ON c.id = cb.chapter_id "
                        "WHERE c.project_id=$1 AND c.idx<$2 "
                        "  AND cb.search_tsv @@ websearch_to_tsquery('simple', $3) "
                        "ORDER BY ts_rank(cb.search_tsv, websearch_to_tsquery('simple', $3)) DESC "
                        "LIMIT $4",
                        project_id, before_chapter_idx, clean, limit,
                    )
                else:
                    rows = await conn.fetch(
                        "SELECT c.idx, cb.summary, cb.terms_text, cb.facts_text, cb.rules_text "
                        "FROM chapter_briefs cb "
                        "JOIN chapters c ON c.id = cb.chapter_id "
                        "WHERE c.project_id=$1 "
                        "  AND cb.search_tsv @@ websearch_to_tsquery('simple', $2) "
                        "ORDER BY ts_rank(cb.search_tsv, websearch_to_tsquery('simple', $2)) DESC "
                        "LIMIT $3",
                        project_id, clean, limit,
                    )
                for r in rows:
                    text = "\n".join(
                        str(r[k] or "") for k in
                        ("summary", "terms_text", "facts_text", "rules_text")
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
        if scope not in ("all", "translations"):
            return results
        async with self._pool.acquire() as conn:
            for query in queries:
                clean = _clean_query(query)
                if not clean:
                    continue
                rows = await conn.fetch(
                    "SELECT b.source_text, t.translated_text, c.idx, t.page_index "
                    "FROM translations t "
                    "JOIN bubbles b "
                    "  ON b.chapter_id=t.chapter_id "
                    " AND b.page_index=t.page_index "
                    " AND b.bubble_idx=t.bubble_idx "
                    "JOIN chapters c ON c.id = t.chapter_id "
                    "WHERE c.project_id=$1 "
                    "  AND t.translated_text_tsv @@ websearch_to_tsquery('simple', $2) "
                    "ORDER BY ts_rank(t.translated_text_tsv, websearch_to_tsquery('simple', $2)) DESC "
                    "LIMIT $3",
                    project_id, clean, limit,
                )
                for r in rows:
                    h = f"[Ch{r['idx']} p{r['page_index']}] {r['source_text']} → {r['translated_text']}"
                    if h not in seen:
                        seen.add(h)
                        results.append(h)
        return results[:limit]

    async def delete_project(self, project_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM projects WHERE id=$1", project_id)

    async def delete_chapter(self, chapter_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM chapters WHERE id=$1", chapter_id)
        return result.startswith("DELETE ") and not result.endswith("0")

    # ── Queue stats ───────────────────────────────────────────────

    async def queue_stats(self) -> dict:
        async with self._pool.acquire() as conn:
            stage_rows = await conn.fetch(
                "SELECT stage, "
                "  SUM(CASE WHEN claimed_by IS NULL THEN 1 ELSE 0 END) AS pending, "
                "  SUM(CASE WHEN claimed_by IS NOT NULL "
                "       AND claimed_at >= NOW() - INTERVAL '10 minutes' THEN 1 ELSE 0 END) AS running, "
                "  SUM(CASE WHEN claimed_by IS NOT NULL "
                "       AND claimed_at <  NOW() - INTERVAL '10 minutes' THEN 1 ELSE 0 END) AS stale "
                "FROM tasks GROUP BY stage"
            )
            active_rows = await conn.fetch(
                "SELECT DISTINCT claimed_by FROM tasks "
                "WHERE claimed_by IS NOT NULL "
                "  AND claimed_at >= NOW() - INTERVAL '10 minutes'"
            )
        stages: dict[str, dict[str, int]] = {
            r["stage"]: {
                "pending": int(r["pending"] or 0),
                "running": int(r["running"] or 0),
                "stale":   int(r["stale"] or 0),
            }
            for r in stage_rows
        }
        active = [r["claimed_by"] for r in active_rows]
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
    ) -> dict:
        """Find-or-create user from a (provider, external_id) tuple.

        Admin status is derived from the user's Discord role at OAuth
        time (carried in the JWT), not a DB column.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT user_id FROM identities WHERE provider=$1 AND external_id=$2",
                provider, external_id,
            )
            if row:
                user_id = row["user_id"]
                await conn.execute(
                    "UPDATE users SET display_name=$1, avatar_url=$2, "
                    "  email=COALESCE($3, email), last_login_at=NOW() "
                    "WHERE id=$4",
                    display_name, avatar_url, email, user_id,
                )
                if metadata is not None:
                    await conn.execute(
                        "UPDATE identities SET metadata=$1::jsonb "
                        "WHERE provider=$2 AND external_id=$3",
                        json.dumps(metadata, ensure_ascii=False),
                        provider, external_id,
                    )
            else:
                row = await conn.fetchrow(
                    "INSERT INTO users (display_name, avatar_url, email, last_login_at) "
                    "VALUES ($1,$2,$3, NOW()) RETURNING id",
                    display_name, avatar_url, email,
                )
                user_id = row["id"]
                await conn.execute(
                    "INSERT INTO identities (user_id, provider, external_id, metadata) "
                    "VALUES ($1,$2,$3,$4::jsonb)",
                    user_id, provider, external_id,
                    json.dumps(metadata or {}, ensure_ascii=False),
                )
        user = await self.get_user(user_id)
        assert user is not None
        return user

    async def get_user(self, user_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_USERS} FROM users WHERE id=$1", user_id,
            ))

    async def get_user_by_identity(
        self, provider: str, external_id: str,
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_USERS} FROM users u "
                "JOIN identities i ON i.user_id=u.id "
                "WHERE i.provider=$1 AND i.external_id=$2",
                provider, external_id,
            ))

    async def get_external_id(
        self, user_id: int, provider: str = "discord",
    ) -> str | None:
        """Return the external_id for a user under the given provider."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT external_id FROM identities "
                "WHERE user_id=$1 AND provider=$2 LIMIT 1",
                user_id, provider,
            )
        return row["external_id"] if row else None

    # ── API tokens ────────────────────────────────────────────────

    async def create_api_token(
        self, user_id: int, name: str, prefix: str, token_hash: str,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO api_tokens (user_id, name, prefix, token_hash) "
                "VALUES ($1, $2, $3, $4) RETURNING id",
                user_id, name, prefix, token_hash,
            )
        return row["id"]

    async def list_api_tokens(self, user_id: int) -> list[dict]:
        """Active tokens for a user, newest first. No hash, no plaintext."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, prefix, "
                f"  {_ts('last_used')}, {_ts('created_at')} "
                "FROM api_tokens "
                "WHERE user_id=$1 AND revoked_at IS NULL "
                "ORDER BY id DESC",
                user_id,
            )
        return [dict(r) for r in rows]

    async def candidates_by_prefix(self, prefix: str) -> list[dict]:
        """Return active tokens whose prefix matches.

        Lookup in two steps (prefix narrow + bcrypt verify) — see
        `user_by_api_token` in api/auth_token.py. Prefix is 8 chars
        from a 62-char alphabet → ~218 trillion possibilities, so the
        list is almost always 0 or 1 row.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, user_id, token_hash "
                "FROM api_tokens "
                "WHERE prefix=$1 AND revoked_at IS NULL",
                prefix,
            )
        return [dict(r) for r in rows]

    async def touch_api_token(self, token_id: int) -> None:
        """Bump last_used. Called from auth path; fire-and-forget."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE api_tokens SET last_used=NOW() WHERE id=$1",
                token_id,
            )

    async def revoke_api_token(self, user_id: int, token_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE api_tokens SET revoked_at=NOW() "
                "WHERE id=$1 AND user_id=$2 AND revoked_at IS NULL",
                token_id, user_id,
            )
        return result.startswith("UPDATE ") and not result.endswith("0")

    # ── Events ────────────────────────────────────────────────────

    async def append_event(self, data: dict) -> int:
        """Insert event, return id. Used by EventBus.publish."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO events (data) VALUES ($1::jsonb) RETURNING id",
                json.dumps(data),
            )
        return row["id"]

    async def get_events_after(self, seq: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, data FROM events WHERE id > $1 ORDER BY id LIMIT 100", seq,
            )
        return [{"id": r["id"], **_to_jsonable_dict(r["data"])} for r in rows]

    async def get_chapter_progress(self, chapter_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM events "
                "WHERE data->>'type'='PageDone' "
                "  AND data->>'chapter_id'=$1 "
                "ORDER BY id DESC LIMIT 1",
                str(chapter_id),
            )
        return _to_jsonable_dict(row["data"]) if row else None

    # ── Chapter archive state ─────────────────────────────────────

    async def set_prepared_done(self, chapter_id: int, page_count: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chapters SET page_count=$1, rendered=FALSE WHERE id=$2",
                page_count, chapter_id,
            )

    async def set_rendered(self, chapter_id: int, rendered: bool) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chapters SET rendered=$1 WHERE id=$2",
                rendered, chapter_id,
            )

    async def set_archive(
        self, chapter_id: int, backend: str, locator: str,
    ) -> None:
        """Persist where this chapter's render archive lives.

        Called by the render worker right before set_rendered(True). The
        API uses these to dispatch URL build through the right backend.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chapters SET archive_backend=$1, archive_locator=$2 "
                "WHERE id=$3",
                backend, locator, chapter_id,
            )

    async def get_chapter_render_state(self, chapter_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT rendered, page_count FROM chapters WHERE id=$1",
                chapter_id,
            )
        if row is None:
            return None
        return {"rendered": bool(row["rendered"]), "page_count": row["page_count"]}

    # ── Health ────────────────────────────────────────────────────

    async def ping(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")


# ── Helpers ───────────────────────────────────────────────────────────


async def _verify_schema_version(conn: asyncpg.Connection) -> None:
    """Refuse to start if the schema version doesn't match.

    The events table is BIGSERIAL — restarting against a stale volume
    would keep the sequence going, but other shape changes (column
    drops, type changes) would silently break. Cheaper to fail loud.
    """
    row = await conn.fetchrow(
        "SELECT value FROM meta WHERE key='schema_version'",
    )
    if row is None:
        await conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', $1)",
            SCHEMA_VERSION,
        )
        return
    if row["value"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"schema version mismatch: db has {row['value']!r}, code expects "
            f"{SCHEMA_VERSION!r}. Phase 1 has no migrations — drop and "
            f"recreate the database:  dropdb typoon && createdb -O typoon typoon"
        )


def _to_jsonable(value):
    """asyncpg returns JSONB as `str` in Phase 1 (no codec installed) but
    can return dicts/lists if a codec is set. Normalize both shapes."""
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    return json.loads(value)


def _to_jsonable_dict(value) -> dict:
    out = _to_jsonable(value)
    return out if isinstance(out, dict) else {}


def _derive_state(chapter_row: dict, tasks: list[dict]) -> tuple[str, str, str]:
    """Derive (state, stage, error) from chapter row + tasks list.

    Priority:
      live claim                            → running
      task with attempts ≥ MAX_TASK_ATTEMPTS → error
      stale claim or unclaimed               → pending
      chapters.rendered                      → done
      otherwise                              → idle

    A claim older than STALE_CLAIM_SECONDS is treated as the worker
    having died — the task will be re-claimed by claim_task on the next
    poll, so the UI should show "pending" rather than a misleading
    "running" with no progress.
    """
    now = datetime.now(timezone.utc)

    def is_live_claim(t: dict) -> bool:
        if not t["claimed_by"]:
            return False
        claimed_at = _parse_iso(t.get("claimed_at"))
        if claimed_at is None:
            return True  # unparseable → treat as live, fail safe
        return (now - claimed_at).total_seconds() < STALE_CLAIM_SECONDS

    running = [t for t in tasks if is_live_claim(t)]
    if running:
        return "running", running[0]["stage"], ""
    failed = [t for t in tasks if t["last_error"] and t["attempts"] >= MAX_TASK_ATTEMPTS]
    if failed:
        return "error", failed[0]["stage"], failed[0]["last_error"] or ""
    pending = [t for t in tasks]  # everything left = unclaimed or stale-claim
    if pending:
        return "pending", pending[0]["stage"], ""
    if chapter_row.get("rendered"):
        return "done", "", ""
    return "idle", "", ""


def _parse_iso(value) -> datetime | None:
    """Parse the API's RFC 3339 string back to an aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
