"""Postgres storage — material/translation entities + pipeline coordination.

Schema lives in `schema.sql` and is applied idempotently at `open()`;
there is no migration tooling — bump SCHEMA_VERSION + drop/recreate the
database during dev when shape changes.

Datetime handling: asyncpg returns `datetime` for TIMESTAMPTZ. We
convert each timestamp to RFC 3339 in UTC at the SQL layer
(`to_char(... AT TIME ZONE 'UTC', ... )`) so callers see strings only.

See docs/rfc/material-architecture.md for the design rationale.
"""

from __future__ import annotations

import json
import logging
import re as _re
from typing import TYPE_CHECKING, Literal

import asyncpg

if TYPE_CHECKING:
    from typoon.adapters.inbox import InboxHandle

logger = logging.getLogger(__name__)

# Bump when schema.sql changes shape. Mismatch on boot ⇒ refuse to
# start, instruct the operator to nuke the volume.
SCHEMA_VERSION = "24"  # library_entries.target_lang — per-entry reading language preference

# Hard cap on retry attempts per task. Deterministic crashes (NameError,
# malformed input, persistent OOM) must not loop forever — after this
# many failures the task is dead-lettered: visible to status views
# (last_error populated) but never re-claimed until an operator redoes.
MAX_TASK_ATTEMPTS = 3

# How long a claim is considered "live". After this, the task is
# re-claimable by another worker AND status views treat it as pending
# rather than running. Without this, a worker that crashes hard (OOM,
# killed) would leave its task forever stuck on "running" in the UI.
STALE_CLAIM_SECONDS = 10 * 60

# Sparse server-managed sort key for chapters. New rows land INITIAL_GAP
# apart on append; midpoint bisect on insert keeps it dense-ish without
# rewriting siblings. Full rebalance fires only when adjacent neighbours
# sit fewer than REBALANCE_MIN_GAP apart — a corner case real manga
# insertion patterns (mostly append, occasional Extra/Volume cover) do
# not usually reach.
INITIAL_GAP        = 1024
REBALANCE_MIN_GAP  = 2


# ── Helpers ───────────────────────────────────────────────────────────


def _read_schema_sql() -> str:
    from pathlib import Path
    return (Path(__file__).parent / "schema.sql").read_text()


# Postgres FTS query sanitiser. The agent passes natural strings
# (`"phép thuật"`, `magic OR sorcery`, `-cấm`) through search endpoints.
# `websearch_to_tsquery` accepts Google-style syntax directly — no
# escaping needed. We only strip control chars asyncpg refuses.
_CTRL = _re.compile(r"[\x00-\x1f]")


def _clean_query(q: str) -> str:
    return _CTRL.sub(" ", q).strip()


# ISO-string timestamp formatting. Convert all timestamps to UTC and
# format as RFC 3339 with `Z` suffix — directly parseable by JS
# `new Date(...)` and avoids local-time-zone surprises across hosts.
_ISO_FMT = "YYYY-MM-DD\"T\"HH24:MI:SS\"Z\""


def _ts(col: str, alias: str | None = None) -> str:
    """Render a `to_char(... AT TIME ZONE 'UTC') AS <alias>` clause for
    an ISO-8601 timestamp column.

    When `col` is qualified (e.g. `t.created_at`), `alias` MUST be
    supplied — Postgres rejects dotted identifiers in `AS …`. Plain
    unqualified columns default the alias to the column name.
    """
    if alias is None:
        if "." in col:
            raise ValueError(
                f"_ts: qualified column {col!r} requires explicit alias"
            )
        alias = col
    return f"to_char(({col}) AT TIME ZONE 'UTC', '{_ISO_FMT}') AS {alias}"


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection init hook for the pool.

    Registers a JSONB codec so every JSON-shaped column comes back as a
    Python object (dict/list) rather than the raw string. Without this
    asyncpg returns JSONB as `str` whenever `statement_cache_size=0`
    bypasses its type introspection — which used to force every read
    site to defensively `json.loads(...)` on a `isinstance(v, str)`
    branch. One codec, one source of truth.
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda v: json.dumps(v, ensure_ascii=False),
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=lambda v: json.dumps(v, ensure_ascii=False),
        decoder=json.loads,
        schema="pg_catalog",
    )


_TS_USERS = (
    "id, display_name, avatar_url, email, "
    f"{_ts('created_at')}, {_ts('last_login_at')}"
)
_TS_MATERIAL = (
    "id, imported_by, origin, work_id, source, upstream_ref, title, cover_url, "
    "description, author, status, languages, title_native, title_alt, "
    "cross_refs, nsfw, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_CHAPTER = (
    "id, material_id, work_chapter_id, position, label, upstream_url, "
    "source_lang, "
    "prepared_hash, prepared_backend, prepared_locator, "
    "masks_backend, masks_locator, page_count, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
# Same column list as `_TS_CHAPTER` but qualified with the `c` alias
# so it can sit beside `wc.*` columns without an `id` collision.
_TS_CHAPTER_C = (
    "c.id, c.material_id, c.work_chapter_id, c.position, c.label, "
    "c.upstream_url, c.source_lang, "
    "c.prepared_hash, c.prepared_backend, c.prepared_locator, "
    "c.masks_backend, c.masks_locator, c.page_count, "
    f"{_ts('c.created_at', 'created_at')}, "
    f"{_ts('c.updated_at', 'updated_at')}"
)
_TS_DRAFT = (
    "id, chapter_id, source_lang, target_lang, glossary_fp, llm_model, "
    "created_by, state, error_message, "
    "progress_stage, progress_index, progress_total, "
    "archive_backend, archive_locator, "
    f"{_ts('rendered_at')}, "
    f"{_ts('takedown_at')}, takedown_reason, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_TRANSLATION = (
    "id, work_chapter_id, owner_id, target_lang, draft_id, shared, "
    "archive_backend, archive_locator, "
    f"{_ts('rendered_at')}, "
    f"{_ts('takedown_at')}, takedown_reason, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_LIBRARY_ENTRY = (
    "id, user_id, work_id, title, cover_url, target_lang, status, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_TRANSLATOR_MEMORY = (
    "id, user_id, material_id, source_lang, target_lang, "
    "characters, world, style, glossary, style_refs, last_chapter_id, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)


def _row_dict(row: asyncpg.Record | None) -> dict | None:
    return dict(row) if row else None


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


# Chapter `progress_stage/index/total` are also columns on
# translation_drafts. Define progress dict shape once.
def _progress_or_none(row: dict) -> dict | None:
    if row.get("progress_stage") is None:
        return None
    return {
        "stage": row["progress_stage"],
        "index": int(row.get("progress_index") or 0),
        "total": int(row.get("progress_total") or 0),
    }


# ── Chapter position assignment ───────────────────────────────────────


async def _resolve_chapter_position(
    conn: asyncpg.Connection, material_id: int, number_norm: str,
) -> int:
    """Compute the `position` for a new chapter being inserted.

    Caller must hold the material's advisory xact lock.

    Sort key is the canonical `number_norm` from the chapter's
    `work_chapter`, joined back via `chapters.work_chapter_id`. Using
    the canonical key means "Chương 040" and "Chapter 40" land at
    the same position across sibling materials.

    Strategy:
      - "extra" / "oneshot" / anything not parseable as a float
        → append at max(position) + INITIAL_GAP.
      - Numeric norm → place between the last sibling ≤ target and
        the first sibling > target. Equality goes to `prev` so a
        later upload of the same chapter lands AFTER the existing
        one (first-come stays first).
      - When the chosen gap is below REBALANCE_MIN_GAP, redistribute
        the whole material to INITIAL_GAP spacing and retry once.
    """
    target = _try_float(number_norm)
    rows = await conn.fetch(
        "SELECT c.position, wc.number_norm "
        "FROM chapters c "
        "JOIN work_chapters wc ON wc.id = c.work_chapter_id "
        "WHERE c.material_id=$1 ORDER BY c.position",
        material_id,
    )
    if not rows:
        return INITIAL_GAP

    if target is None:
        return rows[-1]["position"] + INITIAL_GAP

    prev_pos: int | None = None
    next_pos: int | None = None
    for r in rows:
        n = _try_float(r["number_norm"])
        if n is None:
            continue  # non-numeric siblings ignored for ordering
        if n <= target:
            prev_pos = r["position"]
        elif next_pos is None:
            next_pos = r["position"]
            break

    if prev_pos is None and next_pos is None:
        return rows[-1]["position"] + INITIAL_GAP
    if prev_pos is None:
        return next_pos - INITIAL_GAP  # type: ignore[operator]
    if next_pos is None:
        return prev_pos + INITIAL_GAP

    if next_pos - prev_pos < REBALANCE_MIN_GAP:
        await _rebalance_positions(conn, material_id)
        return await _resolve_chapter_position(conn, material_id, number_norm)
    return (prev_pos + next_pos) // 2


async def _rebalance_positions(
    conn: asyncpg.Connection, material_id: int,
) -> None:
    """Redistribute every chapter's position to INITIAL_GAP spacing.

    Called only when bisect runs out of room between two siblings — a
    pathological case for manga workloads but tolerable as a single
    UPDATE within the upload transaction.
    """
    await conn.execute(
        """
        UPDATE chapters AS c
        SET    position = ranked.new_pos
        FROM (
            SELECT id,
                   ROW_NUMBER() OVER (ORDER BY position) * $2 AS new_pos
            FROM   chapters
            WHERE  material_id = $1
        ) AS ranked
        WHERE c.id = ranked.id
        """,
        material_id, INITIAL_GAP,
    )


# Canonical ordering for (a, b) pairs in material_link_votes —
# CHECK constraint enforces a < b, callers should not need to know.
def _canonical_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _cross_refs_conflict(a: dict, b: dict) -> bool:
    """Mirror `typoon.adapters.work_linker._is_conflict` for the
    inline merge path. Two cross_refs maps conflict when any shared
    namespace carries different stringified values; disjoint
    namespaces never conflict.
    """
    if not a or not b:
        return False
    for k, va in a.items():
        if k in b:
            vb = b[k]
            if va is None or vb is None:
                continue
            if str(va).strip() != str(vb).strip():
                return True
    return False


async def _merge_works(
    conn: asyncpg.Connection, *, canonical: int, doomed: int,
) -> None:
    """Inline community-vote merge: dissolve `doomed` into `canonical`.

    Steps (all inside the caller's transaction):

      1. Re-point every material currently attached to `doomed` to
         `canonical`. UNIQUE (user, work) on `library_entries` is
         work-scoped, so two entries could now collide — defer that
         dedup to step 4.
      2. For every `work_chapters` row on `doomed`:
           - if `canonical` already has a row at the same
             `number_norm`, re-point dependent rows (chapters,
             translations, reading_history) to canonical's row and
             DELETE the doomed row;
           - else move the row to canonical via UPDATE work_id.
      3. Merge cross_refs (additive — canonical wins on shared keys
         because `_cross_refs_conflict` was already checked).
      4. Collapse duplicate library_entries (user_id, work_id):
         keep the older row, re-point library_materials, drop the
         dup. Preserves the user's earliest "Theo dõi" timestamp.
      5. DELETE the now-empty `works` row.
    """
    # 1. Move materials.
    await conn.execute(
        "UPDATE materials SET work_id=$1 WHERE work_id=$2",
        canonical, doomed,
    )

    # 2. Reconcile work_chapters one-by-one.
    doomed_chs = await conn.fetch(
        "SELECT id, number_norm FROM work_chapters WHERE work_id=$1",
        doomed,
    )
    for r in doomed_chs:
        doomed_wc = int(r["id"])
        norm      = r["number_norm"]
        existing = await conn.fetchrow(
            "SELECT id FROM work_chapters "
            "WHERE work_id=$1 AND number_norm=$2",
            canonical, norm,
        )
        if existing is None:
            await conn.execute(
                "UPDATE work_chapters SET work_id=$1 WHERE id=$2",
                canonical, doomed_wc,
            )
            continue
        target_wc = int(existing["id"])
        await conn.execute(
            "UPDATE chapters SET work_chapter_id=$1 WHERE work_chapter_id=$2",
            target_wc, doomed_wc,
        )
        # translations.work_chapter_id may collide on UNIQUE
        # (work_chapter_id, owner_id, target_lang). Delete the colliding
        # duplicate after the re-point — canonical's translation wins
        # (older row stays). Acceptable trade-off: vote-driven merge
        # is irreversible from the user's POV anyway.
        dups = await conn.fetch(
            """
            DELETE FROM translations
            WHERE work_chapter_id=$1
              AND (owner_id, target_lang) IN (
                SELECT owner_id, target_lang
                FROM translations
                WHERE work_chapter_id=$2
              )
            RETURNING id
            """,
            doomed_wc, target_wc,
        )
        _ = dups  # discard — translations cascade their archive locators on later GC.
        await conn.execute(
            "UPDATE translations SET work_chapter_id=$1 WHERE work_chapter_id=$2",
            target_wc, doomed_wc,
        )
        # reading_history collides on PRIMARY KEY (user_id, work_chapter_id).
        # Same strategy: drop the doomed-side rows whose user already
        # has a row on the canonical chapter, then re-point the rest.
        await conn.execute(
            """
            DELETE FROM reading_history
            WHERE work_chapter_id=$1
              AND user_id IN (
                SELECT user_id FROM reading_history
                WHERE work_chapter_id=$2
              )
            """,
            doomed_wc, target_wc,
        )
        await conn.execute(
            "UPDATE reading_history SET work_chapter_id=$1 "
            "WHERE work_chapter_id=$2",
            target_wc, doomed_wc,
        )
        # work_chapter row now empty.
        await conn.execute(
            "DELETE FROM work_chapters WHERE id=$1", doomed_wc,
        )

    # 3. Merge cross_refs additively — canonical's values on shared
    # keys are preserved (`||` is right-takes-precedence, so doomed's
    # map sits on the left, canonical's on the right).
    await conn.execute(
        "UPDATE works SET "
        "  cross_refs = COALESCE((SELECT cross_refs FROM works WHERE id=$2), '{}'::jsonb) "
        "             || COALESCE(cross_refs, '{}'::jsonb), "
        "  updated_at = NOW() "
        "WHERE id=$1",
        canonical, doomed,
    )

    # 4. Collapse duplicate library_entries.
    dup_entries = await conn.fetch(
        """
        SELECT user_id, ARRAY_AGG(id ORDER BY id ASC) AS ids
        FROM library_entries
        WHERE work_id=$1
        GROUP BY user_id
        HAVING COUNT(*) > 1
        """,
        canonical,
    )
    for r in dup_entries:
        ids = [int(x) for x in r["ids"]]
        keeper = ids[0]
        losers = ids[1:]
        await conn.execute(
            "UPDATE library_materials SET entry_id=$1 "
            "WHERE entry_id = ANY($2::bigint[])",
            keeper, losers,
        )
        await conn.execute(
            "DELETE FROM library_entries WHERE id = ANY($1::bigint[])",
            losers,
        )

    # 5. Log the redirect BEFORE dropping the doomed row — clients
    # holding `/w/<doomed>` URLs (open tabs, shared links) hit the
    # work_redirects table on their next GET and learn the canonical
    # id without seeing a 404. The schema-level trigger
    # `work_redirects_collapse` rewrites any older redirect that
    # pointed at `doomed` to point at `canonical` instead, so
    # multi-step merges (A→B then B→C) don't force clients through
    # extra hops.
    await conn.execute(
        "INSERT INTO work_redirects (old_id, new_id) VALUES ($1, $2) "
        "ON CONFLICT (old_id) DO UPDATE SET new_id = EXCLUDED.new_id, "
        "  merged_at = NOW()",
        doomed, canonical,
    )

    # 6. Drop the doomed work. `work_redirects.new_id` has a
    # CASCADE FK, but the only row referencing `doomed` after the
    # trigger collapse is the one we just inserted (whose new_id is
    # `canonical`, not `doomed`), so CASCADE removes nothing extra.
    await conn.execute("DELETE FROM works WHERE id=$1", doomed)


async def _verify_schema_version(conn: asyncpg.Connection) -> None:
    row = await conn.fetchrow(
        "SELECT value FROM meta WHERE key='schema_version'",
    )
    if row is None:
        # Fresh DB. Stamp the current version + apply schema.
        await conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', $1)",
            SCHEMA_VERSION,
        )
        return
    if row["value"] != SCHEMA_VERSION:
        raise RuntimeError(
            f"DB schema version {row['value']!r} != expected "
            f"{SCHEMA_VERSION!r}. Drop and recreate the DB:\n\n"
            "    dropdb typoon && createdb -O typoon typoon\n"
        )


# ── PostgresStore ─────────────────────────────────────────────────────


class PostgresStore:
    """Concrete Store backed by Postgres 17. See `typoon/storage/store.py`
    for the protocol; this class is the only production implementation."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @staticmethod
    async def open(
        dsn: str,
        *,
        pool_min_size:        int = 2,
        pool_max_size:        int = 10,
        statement_cache_size: int = 0,
    ) -> "PostgresStore":
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"DATABASE_URL must be a postgresql:// DSN, got: {dsn!r}"
            )
        pool = await asyncpg.create_pool(
            dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
            statement_cache_size=statement_cache_size,
            init=_init_connection,
        )
        async with pool.acquire() as conn:
            # Verify schema version BEFORE applying schema.sql — a stale
            # volume must fail loud with the dropdb hint instead of
            # crashing on `CREATE INDEX` against a missing column. The
            # meta table is the only DDL we need before shape check.
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS meta ("
                "  key TEXT PRIMARY KEY, value TEXT NOT NULL"
                ")"
            )
            await _verify_schema_version(conn)
            await conn.execute(_read_schema_sql())
        return PostgresStore(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def ping(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

    # ── Meta key/value (singletons: schema version, guild branding) ─

    async def get_meta(self, key: str) -> str | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT value FROM meta WHERE key=$1", key,
            )

    async def set_meta(self, key: str, value: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO meta (key, value) VALUES ($1, $2) "
                "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                key, value,
            )

    # ── Identity ──────────────────────────────────────────────────

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
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT user_id FROM identities "
                "WHERE provider=$1 AND external_id=$2",
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
                        metadata,
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
                    metadata or {},
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
        *, scopes: list[str] | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO api_tokens (user_id, name, prefix, token_hash, scopes) "
                "VALUES ($1, $2, $3, $4, $5) RETURNING id",
                user_id, name, prefix, token_hash, list(scopes or []),
            )
        return row["id"]

    async def list_api_tokens(self, user_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, prefix, scopes, "
                f"  {_ts('last_used')}, {_ts('created_at')} "
                "FROM api_tokens "
                "WHERE user_id=$1 AND revoked_at IS NULL "
                "ORDER BY id DESC",
                user_id,
            )
        return [dict(r) for r in rows]

    async def candidates_by_prefix(self, prefix: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, user_id, token_hash, scopes "
                "FROM api_tokens "
                "WHERE prefix=$1 AND revoked_at IS NULL",
                prefix,
            )
        return [dict(r) for r in rows]

    async def touch_api_token(self, token_id: int) -> None:
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

    # ── Work (global identity) ────────────────────────────────────

    async def get_work(self, work_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                "SELECT id, cross_refs, "
                f"{_ts('created_at')}, {_ts('updated_at')} "
                "FROM works WHERE id=$1",
                work_id,
            ))

    async def get_work_redirect(self, old_id: int) -> int | None:
        """Resolve a dissolved Work id to its canonical replacement.

        Returns the live `works.id` the client should switch to, or
        `None` when `old_id` was never merged (either it's still a
        valid Work or it never existed). Transitive merges are
        already collapsed on insert (see `collapse_work_redirects`
        trigger), so one lookup is enough.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT new_id FROM work_redirects WHERE old_id=$1",
                old_id,
            )
        return int(row["new_id"]) if row else None

    async def find_or_create_work_chapter(
        self,
        *,
        work_id:     int,
        number_norm: str,
        label:       str | None = None,
    ) -> int:
        """Idempotent insert keyed on (work_id, number_norm).

        Advisory xact lock on the work scopes serialization so two
        concurrent spawns of the same (work, chapter) collapse to one
        row. `label` is first-write-wins; updates after creation are
        a no-op here (admin can rename via a dedicated path).
        """
        async with self._pool.acquire() as conn, conn.transaction():
            # Negative offset distinguishes work locks from material
            # locks (`_resolve_chapter_position` already uses positive
            # material_id values for its own advisory_xact_lock).
            await conn.execute(
                "SELECT pg_advisory_xact_lock($1)", -work_id,
            )
            existing = await conn.fetchrow(
                "SELECT id FROM work_chapters "
                "WHERE work_id=$1 AND number_norm=$2",
                work_id, number_norm,
            )
            if existing is not None:
                return int(existing["id"])
            row = await conn.fetchrow(
                "INSERT INTO work_chapters (work_id, number_norm, label) "
                "VALUES ($1, $2, $3) RETURNING id",
                work_id, number_norm, label,
            )
            return int(row["id"])

    async def get_work_chapter(self, work_chapter_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                "SELECT id, work_id, number_norm, label, "
                f"{_ts('created_at')} "
                "FROM work_chapters WHERE id=$1",
                work_chapter_id,
            ))

    async def list_materials_for_work(self, work_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_TS_MATERIAL} FROM materials "
                "WHERE work_id=$1 ORDER BY created_at ASC, id ASC",
                work_id,
            )
        return [dict(r) for r in rows]

    async def list_work_chapters_with_translations(
        self,
        work_id:   int,
        *,
        viewer_id: int,
    ) -> list[dict]:
        """Single-query overlay: every work_chapter of the work plus
        each shared / viewer-owned translation, ordered by chapter
        number_norm desc (latest first, NULLS LAST behaviour) and
        translation created_at desc.

        Output rows aggregated by chapter in Python so the SQL stays
        flat — translations array assembled below.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    wc.id            AS work_chapter_id,
                    wc.number_norm,
                    wc.label,
                    wc.created_at    AS wc_created_at,
                    t.id             AS translation_id,
                    t.target_lang,
                    t.owner_id,
                    t.shared,
                    t.archive_locator IS NULL AND t.draft_id IS NOT NULL
                                     AS uses_default_render,
                    COALESCE(d.state, 'done') AS state,
                    d.error_message  AS error_message,
                    t.draft_id,
                    d.source_lang    AS draft_source_lang,
                    d.chapter_id     AS draft_chapter_id,
                    c.material_id    AS draft_material_id,
                    u.display_name   AS creator_name,
                    to_char(t.updated_at AT TIME ZONE 'UTC',
                            'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS t_updated_at
                FROM work_chapters wc
                LEFT JOIN translations t
                  ON t.work_chapter_id = wc.id
                 AND t.takedown_at IS NULL
                 AND (t.shared = TRUE OR t.owner_id = $2)
                LEFT JOIN translation_drafts d ON d.id = t.draft_id
                LEFT JOIN chapters c ON c.id = d.chapter_id
                LEFT JOIN users u ON u.id = t.owner_id
                WHERE wc.work_id = $1
                ORDER BY wc.id, t.created_at DESC
                """,
                work_id, viewer_id,
            )

        # Group translations under their work_chapter, preserving the
        # SQL order (newest translation first per chapter).
        by_wc: dict[int, dict] = {}
        for r in rows:
            wc_id = int(r["work_chapter_id"])
            chapter = by_wc.get(wc_id)
            if chapter is None:
                chapter = {
                    "id":          wc_id,
                    "number_norm": r["number_norm"],
                    "label":       r["label"],
                    "translations": [],
                }
                by_wc[wc_id] = chapter
            if r["translation_id"] is None:
                continue
            chapter["translations"].append({
                "id":                  int(r["translation_id"]),
                "target_lang":         r["target_lang"],
                "source_lang":         r.get("draft_source_lang"),
                "owner_id":            int(r["owner_id"]),
                "creator_name":        r.get("creator_name"),
                "state":               r["state"],
                "error_message":       r.get("error_message"),
                "shared":              bool(r["shared"]),
                "draft_id":            int(r["draft_id"]) if r["draft_id"] else None,
                "draft_chapter_id":    int(r["draft_chapter_id"]) if r["draft_chapter_id"] else None,
                "draft_material_id":   int(r["draft_material_id"]) if r["draft_material_id"] else None,
                "uses_default_render": bool(r["uses_default_render"]),
                "updated_at":          r.get("t_updated_at"),
            })
        # Sort chapters latest-first by number_norm (numeric where it
        # parses, lexicographic fallback for non-numeric like "extra").
        def _sortkey(c: dict) -> tuple[int, float, str]:
            n = _try_float(c["number_norm"] or "")
            if n is None:
                return (0, 0.0, c["number_norm"] or "")
            return (1, n, "")
        return sorted(by_wc.values(), key=_sortkey, reverse=True)

    # ── Material ──────────────────────────────────────────────────

    async def get_or_create_source_material(
        self,
        *,
        source:       str,
        upstream_ref: str,
        title:        str,
        cover_url:    str | None = None,
        description:  str | None = None,
        author:       str | None = None,
        status:       str | None = None,
        languages:    list[str] | None = None,
        title_native: str | None = None,
        title_alt:    list[str] | None = None,
        cross_refs:   dict | None = None,
        nsfw:         bool = False,
        imported_by:  int | None = None,
    ) -> int:
        """Cross-user dedup on (source, upstream_ref). Display snapshot
        is first-write-wins; ON CONFLICT DO NOTHING leaves existing
        rows untouched. A later background job may refresh metadata.

        Auto-links the material to its canonical Work in the same
        transaction (see :mod:`typoon.adapters.work_linker`). When a
        row already exists we keep its current `work_id` and only
        merge in any new cross_refs namespaces the caller brought
        along.
        """
        from typoon.adapters.work_linker import link_or_create_work

        async with self._pool.acquire() as conn, conn.transaction():
            existing = await conn.fetchrow(
                "SELECT id, work_id FROM materials "
                "WHERE source=$1 AND upstream_ref=$2",
                source, upstream_ref,
            )
            if existing is not None:
                # Augment the existing Work with any new cross_refs
                # the caller surfaced. Keeps the row's work_id stable.
                if cross_refs:
                    from typoon.adapters.work_linker import _merge_refs, _clean_refs
                    refs = _clean_refs(cross_refs)
                    if refs:
                        await _merge_refs(conn, int(existing["work_id"]), refs)
                return int(existing["id"])

            work_id = await link_or_create_work(conn, cross_refs)
            row = await conn.fetchrow(
                "INSERT INTO materials ("
                "  imported_by, origin, work_id, source, upstream_ref, "
                "  title, cover_url, description, author, status, "
                "  languages, title_native, title_alt, cross_refs, nsfw"
                ") VALUES ($1, 'source', $2, $3, $4, $5, $6, $7, $8, $9, "
                "          $10, $11, $12, $13::jsonb, $14) "
                "RETURNING id",
                imported_by, work_id, source, upstream_ref, title, cover_url,
                description, author, status,
                list(languages or []), title_native, list(title_alt or []),
                cross_refs if cross_refs else None, nsfw,
            )
        return row["id"]

    async def create_local_material(
        self,
        *,
        origin:       str,  # 'extension' | 'upload'
        title:        str,
        cover_url:    str | None = None,
        description:  str | None = None,
        author:       str | None = None,
        nsfw:         bool = False,
        imported_by:  int | None = None,
    ) -> int:
        if origin not in ("extension", "upload"):
            raise ValueError(f"create_local_material origin invalid: {origin!r}")
        from typoon.adapters.work_linker import link_or_create_work

        async with self._pool.acquire() as conn, conn.transaction():
            # Ext / upload have no cross_refs at creation — every
            # row gets an isolated Work, link votes can merge later.
            work_id = await link_or_create_work(conn, None)
            row = await conn.fetchrow(
                "INSERT INTO materials ("
                "  imported_by, origin, work_id, title, cover_url, description, "
                "  author, nsfw"
                ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id",
                imported_by, origin, work_id, title, cover_url,
                description, author, nsfw,
            )
        return row["id"]

    async def get_material(self, material_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_MATERIAL} FROM materials WHERE id=$1",
                material_id,
            ))

    async def update_material_metadata(
        self, material_id: int,
        *,
        title:        str | None = None,
        cover_url:    str | None = None,
        description:  str | None = None,
        nsfw:         bool | None = None,
    ) -> None:
        sets: list[str] = []
        args: list = []
        if title is not None:
            args.append(title);       sets.append(f"title=${len(args)}")
        if cover_url is not None:
            args.append(cover_url);   sets.append(f"cover_url=${len(args)}")
        if description is not None:
            args.append(description); sets.append(f"description=${len(args)}")
        if nsfw is not None:
            args.append(nsfw);        sets.append(f"nsfw=${len(args)}")
        if not sets:
            return
        args.append(material_id)
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE materials SET {', '.join(sets)} WHERE id=${len(args)}",
                *args,
            )

    async def delete_material(self, material_id: int) -> None:
        """Cascades through chapters → drafts/translations → bubbles
        etc. via FK ON DELETE CASCADE. Blob cleanup is the caller's
        job (route layer queries chapter locators before delete)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM materials WHERE id=$1", material_id,
            )

    # ── Cross-source linking (community-voted) ────────────────────

    async def cast_material_link_vote(
        self, voter_id: int, material_a_id: int, material_b_id: int,
        vote: int,
    ) -> None:
        if vote not in (-1, 1):
            raise ValueError(f"vote must be ±1, got {vote}")
        a, b = _canonical_pair(material_a_id, material_b_id)
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO material_link_votes "
                "  (material_a_id, material_b_id, voter_id, vote) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (material_a_id, material_b_id, voter_id) DO UPDATE "
                "  SET vote=EXCLUDED.vote, voted_at=NOW()",
                a, b, voter_id, vote,
            )

    async def remove_material_link_vote(
        self, voter_id: int, material_a_id: int, material_b_id: int,
    ) -> None:
        a, b = _canonical_pair(material_a_id, material_b_id)
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM material_link_votes "
                "WHERE material_a_id=$1 AND material_b_id=$2 AND voter_id=$3",
                a, b, voter_id,
            )

    async def get_material_link_score(
        self, material_a_id: int, material_b_id: int,
    ) -> tuple[int, int]:
        a, b = _canonical_pair(material_a_id, material_b_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                # Live aggregation — materialized view may lag.
                "SELECT COALESCE(SUM(vote), 0)::INTEGER AS score, "
                "       COUNT(*)::INTEGER AS total "
                "FROM material_link_votes "
                "WHERE material_a_id=$1 AND material_b_id=$2",
                a, b,
            )
        return (int(row["score"] or 0), int(row["total"] or 0))

    async def refresh_material_links(self) -> None:
        async with self._pool.acquire() as conn:
            # CONCURRENTLY requires the unique index we created;
            # falls back to non-concurrent if it's the first refresh.
            try:
                await conn.execute(
                    "REFRESH MATERIALIZED VIEW CONCURRENTLY material_links"
                )
            except asyncpg.exceptions.FeatureNotSupportedError:
                await conn.execute("REFRESH MATERIALIZED VIEW material_links")

    async def cast_link_vote_with_merge(
        self,
        *,
        voter_id:     int,
        material_a_id: int,
        material_b_id: int,
        vote:         int,
        threshold:    int = 3,
    ) -> dict:
        """Cast a +1/-1 link vote on (a, b). If the resulting score
        crosses `threshold` AND the two materials are still in
        different Works AND their Works don't have conflicting
        cross_refs, merge the newer Work into the older one inline.

        Returns ``{"vote": ±1, "score": int, "merged": bool,
        "canonical_work_id": int | None, "blocked_reason": str | None}``.

        Conflict signals:
          - 'cross_refs_conflict' — both Works claim the same
            namespace with different values; refuse to merge.
          - 'same_work' — already share a Work (idempotent vote, no
            merge needed).
        """
        if vote not in (-1, 1):
            raise ValueError(f"vote must be ±1, got {vote}")
        if material_a_id == material_b_id:
            raise ValueError("cannot vote on a pair with itself")
        a_id, b_id = _canonical_pair(material_a_id, material_b_id)

        async with self._pool.acquire() as conn, conn.transaction():
            # Cast / upsert the vote first so the new score reflects it.
            await conn.execute(
                "INSERT INTO material_link_votes "
                "  (material_a_id, material_b_id, voter_id, vote) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (material_a_id, material_b_id, voter_id) DO UPDATE "
                "  SET vote=EXCLUDED.vote, voted_at=NOW()",
                a_id, b_id, voter_id, vote,
            )

            score_row = await conn.fetchrow(
                "SELECT COALESCE(SUM(vote), 0)::INTEGER AS score "
                "FROM material_link_votes "
                "WHERE material_a_id=$1 AND material_b_id=$2",
                a_id, b_id,
            )
            score = int(score_row["score"] or 0)

            result: dict = {
                "vote":               vote,
                "score":              score,
                "merged":             False,
                "canonical_work_id":  None,
                "blocked_reason":     None,
            }
            if score < threshold:
                return result

            # Threshold met. Resolve current Works.
            mats = await conn.fetch(
                "SELECT id, work_id FROM materials WHERE id = ANY($1::bigint[])",
                [a_id, b_id],
            )
            by_id = {int(r["id"]): int(r["work_id"]) for r in mats}
            work_a = by_id.get(a_id)
            work_b = by_id.get(b_id)
            if work_a is None or work_b is None:
                # One of the materials vanished — nothing to merge.
                return result
            if work_a == work_b:
                result["canonical_work_id"] = work_a
                result["blocked_reason"]    = "same_work"
                return result

            # Cross-refs conflict check — refuse merge when both Works
            # claim the same namespace with different stringified
            # values.
            refs_rows = await conn.fetch(
                "SELECT id, cross_refs FROM works WHERE id = ANY($1::bigint[])",
                [work_a, work_b],
            )
            refs_by_work = {
                int(r["id"]): (r["cross_refs"] or {}) for r in refs_rows
            }
            if _cross_refs_conflict(refs_by_work.get(work_a) or {},
                                    refs_by_work.get(work_b) or {}):
                result["blocked_reason"] = "cross_refs_conflict"
                return result

            # Pick the older Work as canonical (lower id wins —
            # BIGSERIAL allocates monotonically).
            canonical = min(work_a, work_b)
            doomed    = max(work_a, work_b)
            await _merge_works(conn, canonical=canonical, doomed=doomed)
            result["merged"]             = True
            result["canonical_work_id"]  = canonical
            return result

    async def list_work_link_suggestions(
        self, *, work_id: int,
    ) -> list[dict]:
        """For each (material in this Work) × (material outside this
        Work) pair with a positive vote score, return the suggestion
        the SPA can render in "Manga này ở các nguồn khác".

        Schema 19 keeps votes as a live table; we aggregate at read
        time so a fresh vote in the same session is visible.

        Output (one row per candidate material outside this Work):
            {
              candidate_material_id, candidate_title, candidate_source,
              candidate_cover, candidate_work_id,
              score, total_votes,
              own_material_id  # which sibling triggered the suggestion
            }
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH own_ms AS (
                    SELECT id FROM materials WHERE work_id = $1
                ),
                voted AS (
                    SELECT
                      CASE WHEN v.material_a_id IN (SELECT id FROM own_ms)
                           THEN v.material_b_id
                           ELSE v.material_a_id
                      END AS candidate_id,
                      CASE WHEN v.material_a_id IN (SELECT id FROM own_ms)
                           THEN v.material_a_id
                           ELSE v.material_b_id
                      END AS own_id,
                      v.vote
                    FROM material_link_votes v
                    WHERE v.material_a_id IN (SELECT id FROM own_ms)
                       OR v.material_b_id IN (SELECT id FROM own_ms)
                ),
                agg AS (
                    SELECT candidate_id,
                           SUM(vote)::INTEGER AS score,
                           COUNT(*)::INTEGER AS total,
                           (array_agg(own_id ORDER BY vote DESC))[1] AS own_id
                    FROM voted
                    GROUP BY candidate_id
                )
                SELECT
                    m.id          AS candidate_material_id,
                    m.title       AS candidate_title,
                    m.source      AS candidate_source,
                    m.cover_url   AS candidate_cover,
                    m.work_id     AS candidate_work_id,
                    agg.score, agg.total, agg.own_id AS own_material_id
                FROM agg
                JOIN materials m ON m.id = agg.candidate_id
                WHERE m.work_id != $1
                  AND agg.score > 0
                ORDER BY agg.score DESC, m.id ASC
                """,
                work_id,
            )
        return [dict(r) for r in rows]

    async def list_work_link_candidates(
        self, *, work_id: int, limit: int = 10, threshold: float = 0.45,
    ) -> list[dict]:
        """Title-similarity seed for cross-source link suggestions.

        Pure SQL ranker — no external API. The fanout works as
        follows:

          1. `own` CTE collects every material currently attached to
             this Work, projecting the title strings we want to match
             against (title, title_native, title_alt).
          2. `candidates` CTE pulls every material outside this Work
             that wasn't merged in already and computes the best
             similarity it scored against ANY title from `own`.
          3. We filter out candidates whose Work is already a redirect
             target of (or from) this Work — those are dissolved
             aliases and would just clutter the panel.

        The score lives in 0..1. We boost identical `title_native`
        to 0.95 because cross-source Japanese title agreement is the
        strongest non-cross-ref signal we have (a romanization
        coincidence is far less likely with kanji).

        `pg_trgm`'s `%>` operator uses the GIST index — the query
        runs against a few million materials in <100ms.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH own AS (
                    SELECT id,
                           title,
                           title_native,
                           title_alt,
                           languages
                    FROM materials
                    WHERE work_id = $1
                ),
                own_titles AS (
                    SELECT id AS own_id, UNNEST(
                        ARRAY[title]
                        || COALESCE(ARRAY[title_native]::text[], ARRAY[]::text[])
                        || COALESCE(title_alt, ARRAY[]::text[])
                    ) AS t
                    FROM own
                ),
                pairs AS (
                    SELECT
                      m.id           AS candidate_id,
                      m.title        AS candidate_title,
                      m.title_native AS candidate_native,
                      m.title_alt    AS candidate_alt,
                      m.source       AS candidate_source,
                      m.cover_url    AS candidate_cover,
                      m.work_id      AS candidate_work_id,
                      m.languages    AS candidate_langs,
                      ot.own_id      AS own_id,
                      ot.t           AS own_title,
                      GREATEST(
                          similarity(m.title, ot.t),
                          COALESCE(similarity(m.title_native, ot.t), 0)
                      ) AS sim_base
                    FROM materials m
                    JOIN own_titles ot ON
                        m.title % ot.t
                     OR (m.title_native IS NOT NULL AND m.title_native % ot.t)
                    WHERE m.work_id != $1
                ),
                best AS (
                    SELECT
                        candidate_id,
                        candidate_title,
                        candidate_native,
                        candidate_alt,
                        candidate_source,
                        candidate_cover,
                        candidate_work_id,
                        candidate_langs,
                        MAX(sim_base) AS sim_base,
                        (array_agg(own_id ORDER BY sim_base DESC))[1] AS own_id
                    FROM pairs
                    GROUP BY candidate_id, candidate_title, candidate_native,
                             candidate_alt, candidate_source, candidate_cover,
                             candidate_work_id, candidate_langs
                )
                SELECT
                    b.candidate_id        AS candidate_material_id,
                    b.candidate_title,
                    b.candidate_source,
                    b.candidate_cover,
                    b.candidate_work_id,
                    b.own_id              AS own_material_id,
                    CASE
                      WHEN EXISTS (
                        SELECT 1 FROM own o
                        WHERE o.title_native IS NOT NULL
                          AND b.candidate_native IS NOT NULL
                          AND lower(trim(o.title_native)) = lower(trim(b.candidate_native))
                      ) THEN 0.95
                      WHEN EXISTS (
                        SELECT 1 FROM own o
                        WHERE COALESCE(o.title_alt, ARRAY[]::text[])
                              && COALESCE(b.candidate_alt, ARRAY[]::text[])
                      ) THEN GREATEST(b.sim_base, 0.80)
                      ELSE b.sim_base
                    END AS score,
                    CASE
                      WHEN EXISTS (
                        SELECT 1 FROM own o
                        WHERE o.title_native IS NOT NULL
                          AND b.candidate_native IS NOT NULL
                          AND lower(trim(o.title_native)) = lower(trim(b.candidate_native))
                      ) THEN 'title_native_exact'
                      WHEN EXISTS (
                        SELECT 1 FROM own o
                        WHERE COALESCE(o.title_alt, ARRAY[]::text[])
                              && COALESCE(b.candidate_alt, ARRAY[]::text[])
                      ) THEN 'title_alt_overlap'
                      ELSE 'title_trgm'
                    END AS reason
                FROM best b
                WHERE b.candidate_work_id NOT IN (
                    SELECT new_id FROM work_redirects WHERE old_id = $1
                    UNION ALL
                    SELECT old_id FROM work_redirects WHERE new_id = $1
                )
                  AND (
                    -- Threshold applies to trgm score; native-exact and
                    -- alt-overlap branches always pass since their
                    -- floor is already above the cutoff.
                    b.sim_base >= $2
                    OR EXISTS (
                        SELECT 1 FROM own o
                        WHERE o.title_native IS NOT NULL
                          AND b.candidate_native IS NOT NULL
                          AND lower(trim(o.title_native)) = lower(trim(b.candidate_native))
                    )
                  )
                ORDER BY score DESC, b.candidate_id ASC
                LIMIT $3
                """,
                work_id, threshold, limit,
            )
        return [
            {
                "candidate_material_id": int(r["candidate_material_id"]),
                "candidate_title":       r["candidate_title"],
                "candidate_source":      r["candidate_source"],
                "candidate_cover":       r["candidate_cover"],
                "candidate_work_id":     int(r["candidate_work_id"]),
                "own_material_id":       int(r["own_material_id"]),
                "score":                 float(r["score"]),
                "reason":                r["reason"],
            }
            for r in rows
        ]

    async def get_link_vote(
        self, *, voter_id: int, material_a_id: int, material_b_id: int,
    ) -> int | None:
        """The viewer's own vote on a pair (None when they haven't
        voted). Used by the suggestion UI to show whether the user
        already agreed / rejected."""
        a, b = _canonical_pair(material_a_id, material_b_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT vote FROM material_link_votes "
                "WHERE material_a_id=$1 AND material_b_id=$2 AND voter_id=$3",
                a, b, voter_id,
            )
        return int(row["vote"]) if row else None


    # ── Chapter ───────────────────────────────────────────────────

    async def create_chapter(
        self,
        material_id:  int,
        *,
        number_norm:  str,
        label:        str | None = None,
        upstream_url: str | None = None,
        source_lang:  str | None = None,
    ) -> int:
        """Insert a pixel-bound chapter row. Resolves the material's
        Work id, materialises the matching `work_chapters` row keyed
        on the caller-supplied `number_norm`, then creates the
        chapter under the material's advisory lock.

        `number_norm` is the only chapter-identity field — display
        strings live on `label` (per-source verbatim) and on
        `work_chapters.label` (first-write-wins across sources).
        Position is derived from `number_norm` so sibling materials
        of the same Work sort to the same order.

        `source_lang` is the BCP-47 of the pixels this chapter carries.
        Spawn-translate reads it as the authoritative source language
        instead of falling back to `material.languages[0]`, which is
        wrong whenever a material on MangaDex / Bato hosts chapters
        in mixed languages. NULL is allowed for backwards-compat with
        rows created before schema 26.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            mat = await conn.fetchrow(
                "SELECT work_id, languages FROM materials WHERE id=$1",
                material_id,
            )
            if mat is None:
                raise ValueError(f"create_chapter: material {material_id} not found")
            work_id = int(mat["work_id"])
            # Default the chapter's source_lang from the material's
            # primary language when the caller doesn't override. Keeps
            # the column populated for sources that don't expose a
            # chapter-level language tag.
            if source_lang is None:
                langs = mat.get("languages") or []
                if langs:
                    source_lang = langs[0].lower().split("-")[0]

            # Materialise work_chapter under a work-scoped lock (negative
            # so it can't collide with the material lock below).
            await conn.execute(
                "SELECT pg_advisory_xact_lock($1)", -work_id,
            )
            wc_row = await conn.fetchrow(
                "SELECT id FROM work_chapters "
                "WHERE work_id=$1 AND number_norm=$2",
                work_id, number_norm,
            )
            if wc_row is None:
                wc_row = await conn.fetchrow(
                    "INSERT INTO work_chapters "
                    "  (work_id, number_norm, label) "
                    "VALUES ($1, $2, $3) RETURNING id",
                    work_id, number_norm, label,
                )
            work_chapter_id = int(wc_row["id"])

            # Material-scoped position assignment based on canonical key.
            await conn.execute(
                "SELECT pg_advisory_xact_lock($1)", material_id,
            )
            pos = await _resolve_chapter_position(conn, material_id, number_norm)
            row = await conn.fetchrow(
                "INSERT INTO chapters ("
                "  material_id, work_chapter_id, position, label, "
                "  upstream_url, source_lang"
                ") VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                material_id, work_chapter_id, pos, label, upstream_url,
                source_lang,
            )
        return row["id"]

    async def get_chapter(self, chapter_id: int) -> dict | None:
        """Single chapter row joined with `work_chapters.number_norm`
        so callers see the canonical chapter key without a second
        round-trip — mirrors `list_chapters`."""
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_CHAPTER_C}, wc.number_norm "
                "FROM chapters c "
                "JOIN work_chapters wc ON wc.id = c.work_chapter_id "
                "WHERE c.id=$1",
                chapter_id,
            ))

    async def list_chapters(self, material_id: int) -> list[dict]:
        """Per-material chapter rows ordered by position. Includes
        `number_norm` (joined from `work_chapters`) so callers don't
        need a second round-trip just to read the canonical key —
        this is the closest substitute for the dropped `chapters.number`
        column. `label` carries the source's display string."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_TS_CHAPTER_C}, wc.number_norm "
                "FROM chapters c "
                "JOIN work_chapters wc ON wc.id = c.work_chapter_id "
                "WHERE c.material_id=$1 ORDER BY c.position",
                material_id,
            )
        return [dict(r) for r in rows]

    async def delete_chapter(self, chapter_id: int) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chapters WHERE id=$1", chapter_id,
            )
        return result.startswith("DELETE ") and not result.endswith("0")

    async def set_chapter_prepared(
        self,
        chapter_id: int,
        *,
        prepared_hash: str,
        prepared_backend: str,
        prepared_locator: str,
        page_count: int,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chapters SET "
                "  prepared_hash=$1, prepared_backend=$2, prepared_locator=$3, "
                "  page_count=$4 "
                "WHERE id=$5",
                prepared_hash, prepared_backend, prepared_locator,
                page_count, chapter_id,
            )

    async def set_chapter_masks(
        self, chapter_id: int,
        *, masks_backend: str, masks_locator: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chapters SET masks_backend=$1, masks_locator=$2 "
                "WHERE id=$3",
                masks_backend, masks_locator, chapter_id,
            )

    async def find_chapter_by_prepared_hash(
        self, prepared_hash: str,
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_CHAPTER} FROM chapters "
                "WHERE prepared_hash=$1 AND prepared_locator IS NOT NULL "
                "LIMIT 1",
                prepared_hash,
            ))

    async def find_chapter_by_upstream(
        self, material_id: int, upstream_url: str,
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_CHAPTER} FROM chapters "
                "WHERE material_id=$1 AND upstream_url=$2 LIMIT 1",
                material_id, upstream_url,
            ))

    # ── Scan output (chapter-level) ───────────────────────────────

    async def save_bubbles(
        self, chapter_id: int, bubbles: list[dict],
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM bubbles WHERE chapter_id=$1", chapter_id,
            )
            if not bubbles:
                return
            await conn.executemany(
                "INSERT INTO bubbles "
                "  (chapter_id, page_index, bubble_idx, source_text, "
                "   confidence, shape_kind) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
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
                "SELECT 1 FROM bubbles WHERE chapter_id=$1 LIMIT 1",
                chapter_id,
            )
        return row is not None

    async def save_geometry(
        self, chapter_id: int, pages: list[dict],
    ) -> None:
        """`pages` items: { page_index, width, height, bubbles: [...] }
        where each bubble has bubble_idx, polygon, fit_box, erase_box,
        text_box. Atomic replace per chapter."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM bubble_geometry WHERE chapter_id=$1", chapter_id,
            )
            await conn.execute(
                "DELETE FROM page_geometry WHERE chapter_id=$1", chapter_id,
            )
            for page in pages:
                await conn.execute(
                    "INSERT INTO page_geometry "
                    "  (chapter_id, page_index, width, height) "
                    "VALUES ($1, $2, $3, $4)",
                    chapter_id, page["page_index"],
                    page["width"], page["height"],
                )
                bubbles = page.get("bubbles", [])
                if bubbles:
                    await conn.executemany(
                        "INSERT INTO bubble_geometry "
                        "  (chapter_id, page_index, bubble_idx, "
                        "   polygon, fit_box, erase_box, text_box) "
                        "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, "
                        "        $6::jsonb, $7::jsonb)",
                        [
                            (chapter_id, page["page_index"], b["bubble_idx"],
                             b["polygon"],
                             b["fit_box"],
                             b["erase_box"],
                             b["text_box"])
                            for b in bubbles
                        ],
                    )

    async def get_geometry(self, chapter_id: int) -> list[dict]:
        """Returns one entry per page: { page_index, width, height,
        bubbles: [...] }. Stable order for deterministic render."""
        async with self._pool.acquire() as conn:
            pages = await conn.fetch(
                "SELECT page_index, width, height "
                "FROM page_geometry WHERE chapter_id=$1 "
                "ORDER BY page_index",
                chapter_id,
            )
            bubbles = await conn.fetch(
                "SELECT page_index, bubble_idx, polygon, fit_box, "
                "       erase_box, text_box "
                "FROM bubble_geometry WHERE chapter_id=$1 "
                "ORDER BY page_index, bubble_idx",
                chapter_id,
            )
        by_page: dict[int, list[dict]] = {}
        for b in bubbles:
            d = dict(b)
            by_page.setdefault(d["page_index"], []).append(d)
        return [
            {
                "page_index": p["page_index"],
                "width":      p["width"],
                "height":     p["height"],
                "bubbles":    by_page.get(p["page_index"], []),
            }
            for p in pages
        ]

    # ── Translation drafts (Layer 2) ──────────────────────────────

    async def find_reusable_draft(
        self,
        *,
        chapter_id:  int,
        source_lang: str,
        target_lang: str,
        glossary_fp: str,
    ) -> dict | None:
        """Cache lookup against the global community pool. Returns None
        when no draft matches the key, or the only matches are taken
        down. Schema 19 removed per-guild visibility — every alive
        draft on the cache key is reusable.
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT {_TS_DRAFT}
                FROM   translation_drafts d
                WHERE  d.chapter_id=$1
                  AND  d.source_lang=$2
                  AND  d.target_lang=$3
                  AND  d.glossary_fp=$4
                  AND  d.takedown_at IS NULL
                ORDER BY d.state = 'done' DESC, d.created_at DESC
                LIMIT 1
                """,
                chapter_id, source_lang, target_lang, glossary_fp,
            )
        return _row_dict(row)

    async def create_draft(
        self,
        *,
        chapter_id:  int,
        source_lang: str,
        target_lang: str,
        glossary_fp: str,
        llm_model:   str,
        created_by:  int,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO translation_drafts ("
                "  chapter_id, source_lang, target_lang, glossary_fp, "
                "  llm_model, created_by, state"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'pending') "
                "RETURNING id",
                chapter_id, source_lang, target_lang, glossary_fp,
                llm_model, created_by,
            )
        return row["id"]

    async def get_draft(self, draft_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_DRAFT} FROM translation_drafts WHERE id=$1",
                draft_id,
            ))

    async def update_draft_state(
        self,
        draft_id: int,
        *,
        state:    str,
        error:    str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translation_drafts "
                "SET state=$1, error_message=$2 "
                "WHERE id=$3",
                state, error, draft_id,
            )

    async def set_draft_progress(
        self, draft_id: int, *, stage: str, index: int, total: int,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translation_drafts SET "
                "  progress_stage=$1, progress_index=$2, progress_total=$3 "
                "WHERE id=$4",
                stage, index, total, draft_id,
            )

    async def save_draft_bubbles(
        self, draft_id: int, bubbles: list[dict],
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM translation_draft_bubbles WHERE draft_id=$1",
                draft_id,
            )
            if not bubbles:
                return
            await conn.executemany(
                "INSERT INTO translation_draft_bubbles "
                "  (draft_id, page_index, bubble_idx, translated_text, kind) "
                "VALUES ($1, $2, $3, $4, $5)",
                [
                    (draft_id, b["page_index"], b["bubble_idx"],
                     b["translated_text"], b.get("kind", "dialogue"))
                    for b in bubbles
                ],
            )

    async def get_draft_bubbles(self, draft_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT page_index, bubble_idx, translated_text, kind "
                "FROM translation_draft_bubbles WHERE draft_id=$1 "
                "ORDER BY page_index, bubble_idx",
                draft_id,
            )
        return [dict(r) for r in rows]

    async def save_draft_brief(
        self, draft_id: int, brief: dict,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO draft_briefs ("
                "  draft_id, brief_json, summary, terms_text, "
                "  facts_text, rules_text, updated_at"
                ") VALUES ($1, $2::jsonb, $3, $4, $5, $6, NOW()) "
                "ON CONFLICT (draft_id) DO UPDATE SET "
                "  brief_json=EXCLUDED.brief_json, "
                "  summary=EXCLUDED.summary, "
                "  terms_text=EXCLUDED.terms_text, "
                "  facts_text=EXCLUDED.facts_text, "
                "  rules_text=EXCLUDED.rules_text, "
                "  updated_at=NOW()",
                draft_id, brief,
                brief.get("summary"), brief.get("terms_text"),
                brief.get("facts_text"), brief.get("rules_text"),
            )

    async def get_draft_brief(self, draft_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT brief_json FROM draft_briefs WHERE draft_id=$1",
                draft_id,
            )
        if row is None:
            return None
        return row["brief_json"]

    async def takedown_draft(
        self, draft_id: int, reason: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translation_drafts "
                "SET takedown_at=NOW(), takedown_reason=$2 "
                "WHERE id=$1",
                draft_id, reason,
            )

    async def update_draft_archive(
        self,
        draft_id: int,
        *,
        archive_backend: str,
        archive_locator: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translation_drafts "
                "SET archive_backend=$1, archive_locator=$2, "
                "    rendered_at=NOW() "
                "WHERE id=$3",
                archive_backend, archive_locator, draft_id,
            )

    async def pending_drafts_for_chapter(
        self, chapter_id: int,
    ) -> list[int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id FROM translation_drafts "
                "WHERE chapter_id=$1 "
                "  AND state='pending' "
                "  AND takedown_at IS NULL",
                chapter_id,
            )
        return [r["id"] for r in rows]

    # ── Translations (Layer 3, per-user) ──────────────────────────

    async def get_or_create_translation(
        self,
        *,
        work_chapter_id: int,
        owner_id:        int,
        target_lang:     str,
        draft_id:        int,
        shared:          bool = True,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO translations "
                "  (work_chapter_id, owner_id, target_lang, draft_id, shared) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (work_chapter_id, owner_id, target_lang) DO UPDATE "
                "  SET draft_id = EXCLUDED.draft_id "
                "RETURNING id",
                work_chapter_id, owner_id, target_lang, draft_id, shared,
            )
        return row["id"]

    async def get_translation(self, translation_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_TRANSLATION} FROM translations WHERE id=$1",
                translation_id,
            ))

    async def list_translations_for_chapters(
        self,
        chapter_ids: list[int],
    ) -> dict[int, list[dict]]:
        """Bulk overlay for chapter list views. Joins per-material
        chapter rows to translations via the shared work_chapter so
        a translation spawned from a sibling material of the same
        Work surfaces on every material's chapter list.

        Schema 19 made translations community-wide, so this is a flat
        join — no visibility branch beyond `shared` + takedown.
        """
        if not chapter_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id AS chapter_id,
                    t.id, t.work_chapter_id, t.owner_id, t.target_lang,
                    t.draft_id, t.shared,
                    COALESCE(d.state, 'done') AS state,
                    u.display_name AS creator_name,
                    (t.archive_locator IS NULL AND t.draft_id IS NOT NULL) AS uses_default_render
                FROM chapters c
                JOIN translations t ON t.work_chapter_id = c.work_chapter_id
                LEFT JOIN translation_drafts d ON d.id = t.draft_id
                LEFT JOIN users u ON u.id = t.owner_id
                WHERE c.id = ANY($1::bigint[])
                  AND t.takedown_at IS NULL
                ORDER BY c.id, t.created_at DESC
                """,
                chapter_ids,
            )
        out: dict[int, list[dict]] = {cid: [] for cid in chapter_ids}
        for r in rows:
            d = dict(r)
            cid = d.pop("chapter_id")
            out[cid].append(d)
        return out

    async def list_all_translations_for_chapter(
        self, chapter_id: int,
    ) -> list[dict]:
        """All translations on the Work chapter this pixel chapter
        belongs to. Used by admin cleanup (material delete) to
        enumerate archive locators across owners regardless of
        visibility/takedown."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT t.id, t.owner_id, t.archive_backend, t.archive_locator "
                "FROM translations t "
                "JOIN chapters c ON c.work_chapter_id = t.work_chapter_id "
                "WHERE c.id=$1",
                chapter_id,
            )
        return [dict(r) for r in rows]

    async def list_drafts_for_chapter(
        self, chapter_id: int,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, archive_backend, archive_locator "
                "FROM translation_drafts "
                "WHERE chapter_id=$1",
                chapter_id,
            )
        return [dict(r) for r in rows]


    async def list_my_translations(
        self, user_id: int,
    ) -> list[dict]:
        """Translations owned by `user_id`. The translation row sits at
        Work-chapter scope (cross-source), but for the /translate index
        we surface a single representative material/chapter — the one
        the user spawned from (via translation_drafts.chapter_id).
        Newest activity first.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    t.id              AS translation_id,
                    t.target_lang,
                    {_ts('t.updated_at', 'updated_at')},
                    COALESCE(d.state, 'done') AS state,
                    (t.archive_locator IS NOT NULL
                     OR d.archive_locator IS NOT NULL) AS has_archive,
                    c.id              AS chapter_id,
                    wc.number_norm    AS chapter_number,
                    c.label           AS chapter_label,
                    c.position        AS chapter_position,
                    c.upstream_url    AS chapter_upstream_url,
                    m.id              AS material_id,
                    m.title           AS material_title,
                    m.cover_url       AS material_cover,
                    m.source          AS material_source,
                    m.upstream_ref    AS material_upstream_ref
                FROM translations t
                JOIN translation_drafts d ON d.id = t.draft_id
                JOIN chapters  c ON c.id = d.chapter_id
                JOIN work_chapters wc ON wc.id = c.work_chapter_id
                JOIN materials m ON m.id = c.material_id
                WHERE t.owner_id = $1
                  AND t.takedown_at IS NULL
                ORDER BY t.updated_at DESC
                """,
                user_id,
            )
        return [dict(r) for r in rows]

    async def update_translation_archive(
        self,
        translation_id: int,
        *,
        archive_backend: str,
        archive_locator: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translations "
                "SET archive_backend=$1, archive_locator=$2, rendered_at=NOW() "
                "WHERE id=$3",
                archive_backend, archive_locator, translation_id,
            )

    async def takedown_translation(
        self, translation_id: int, reason: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translations "
                "SET takedown_at=NOW(), takedown_reason=$2 "
                "WHERE id=$1",
                translation_id, reason,
            )

    async def delete_translation(self, translation_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM translations WHERE id=$1",
                translation_id,
            )

    async def upsert_translation_edit(
        self, translation_id: int,
        page_index: int, bubble_idx: int, edited_text: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO translation_edits "
                "  (translation_id, page_index, bubble_idx, edited_text) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (translation_id, page_index, bubble_idx) DO UPDATE "
                "  SET edited_text=EXCLUDED.edited_text, edited_at=NOW()",
                translation_id, page_index, bubble_idx, edited_text,
            )

    async def get_translation_edits(
        self, translation_id: int,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT page_index, bubble_idx, edited_text "
                "FROM translation_edits WHERE translation_id=$1 "
                "ORDER BY page_index, bubble_idx",
                translation_id,
            )
        return [dict(r) for r in rows]

    async def delete_translation_edit(
        self, translation_id: int, page_index: int, bubble_idx: int,
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM translation_edits "
                "WHERE translation_id=$1 AND page_index=$2 AND bubble_idx=$3",
                translation_id, page_index, bubble_idx,
            )
        return result.startswith("DELETE ") and not result.endswith("0")

    # ── Library entries ───────────────────────────────────────────

    async def list_library_entries(
        self,
        user_id: int,
        *,
        status: str | None = None,
    ) -> list[dict]:
        clauses = ["user_id=$1"]
        args: list = [user_id]
        if status is None:
            clauses.append("status<>'dropped'")
        else:
            args.append(status); clauses.append(f"status=${len(args)}")
        async with self._pool.acquire() as conn:
            entries = await conn.fetch(
                f"SELECT {_TS_LIBRARY_ENTRY} FROM library_entries "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY updated_at DESC",
                *args,
            )
            if not entries:
                return []
            ids = [e["id"] for e in entries]
            links = await conn.fetch(
                "SELECT entry_id, material_id, link_origin, "
                f"  {_ts('linked_at')} "
                "FROM library_materials WHERE entry_id = ANY($1::bigint[])",
                ids,
            )
            # Translation summary per entry — pivots by draft state so the
            # library card can render "Đang dịch 2 · Lỗi 1" without N+1
            # queries. Counts only translations the user owns; cross-user
            # cached translations don't count as "the user's work".
            summary_rows = await conn.fetch(
                """
                SELECT lm.entry_id,
                       td.state,
                       COUNT(*) AS n
                FROM   library_materials lm
                JOIN   chapters c          ON c.material_id = lm.material_id
                JOIN   translations t      ON t.work_chapter_id = c.work_chapter_id
                                          AND t.owner_id   = $2
                JOIN   translation_drafts td ON td.id = t.draft_id
                WHERE  lm.entry_id = ANY($1::bigint[])
                  AND  t.takedown_at IS NULL
                GROUP BY lm.entry_id, td.state
                """,
                ids, user_id,
            )
        by_entry: dict[int, list[dict]] = {i: [] for i in ids}
        for l in links:
            d = dict(l)
            eid = d.pop("entry_id")
            by_entry.setdefault(eid, []).append(d)
        summary_by_entry: dict[int, dict[str, int]] = {
            i: {"pending": 0, "running": 0, "done": 0, "error": 0}
            for i in ids
        }
        for r in summary_rows:
            bucket = summary_by_entry.setdefault(
                r["entry_id"],
                {"pending": 0, "running": 0, "done": 0, "error": 0},
            )
            state = r["state"]
            if state in bucket:
                bucket[state] = int(r["n"])
        out: list[dict] = []
        for e in entries:
            d = dict(e)
            d["materials"] = by_entry.get(d["id"], [])
            d["translation_summary"] = summary_by_entry.get(
                d["id"],
                {"pending": 0, "running": 0, "done": 0, "error": 0},
            )
            out.append(d)
        return out

    async def get_library_entry(
        self, entry_id: int, user_id: int,
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            entry = await conn.fetchrow(
                f"SELECT {_TS_LIBRARY_ENTRY} FROM library_entries "
                "WHERE id=$1 AND user_id=$2",
                entry_id, user_id,
            )
            if entry is None:
                return None
            links = await conn.fetch(
                "SELECT material_id, link_origin, "
                f"  {_ts('linked_at')} "
                "FROM library_materials WHERE entry_id=$1",
                entry_id,
            )
            # Same per-state translation summary the list endpoint
            # carries — keeps the hub page's single-entry fetch as
            # informative as the grid's bulk fetch.
            summary_rows = await conn.fetch(
                """
                SELECT td.state, COUNT(*) AS n
                FROM   library_materials lm
                JOIN   chapters c          ON c.material_id = lm.material_id
                JOIN   translations t      ON t.work_chapter_id = c.work_chapter_id
                                          AND t.owner_id   = $2
                JOIN   translation_drafts td ON td.id = t.draft_id
                WHERE  lm.entry_id = $1
                  AND  t.takedown_at IS NULL
                GROUP BY td.state
                """,
                entry_id, user_id,
            )
        d = dict(entry)
        d["materials"] = [dict(l) for l in links]
        summary = {"pending": 0, "running": 0, "done": 0, "error": 0}
        for r in summary_rows:
            state = r["state"]
            if state in summary:
                summary[state] = int(r["n"])
        d["translation_summary"] = summary
        return d

    async def find_entry_for_work(
        self, *, user_id: int, work_id: int,
    ) -> dict | None:
        """Resolve the viewer's library entry for a given Work, if any.
        Used by `/api/work/{id}` to tell the SPA whether to show
        "Theo dõi" or "Mở trong Library" without a second round-trip.
        """
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                "SELECT id, status, target_lang "
                "FROM   library_entries "
                "WHERE  user_id = $1 AND work_id = $2",
                user_id, work_id,
            ))

    async def create_library_entry(
        self,
        *,
        user_id:     int,
        work_id:     int,
        title:       str,
        cover_url:   str | None,
        target_lang: str,
        materials:   list[tuple[int, str]] | None = None,
        status:      str = "reading",
    ) -> int:
        """Insert a (user, work) library entry plus its initial material
        links. `materials` is a list of (material_id, link_origin) the
        caller wants attached at creation time — typically just the
        material the user started from. UNIQUE (user_id, work_id) is
        enforced by the schema; callers must check with
        `find_entry_for_work` first.

        `target_lang` is the user's reading-language preference for
        this Work; the chapter list overlay + manifest fetch (e.g.
        MangaDex multi-language feed) read from it. Defaults are set
        at the route layer, not here.

        Material auto-link votes (used to surface cross-source pairs
        to other users) are NOT cast here; the caller drives those
        explicitly via `link_material_to_entry` if needed. The first
        material on a brand-new entry has no peers to vote on.
        """
        if status not in ("reading", "plan", "done", "dropped"):
            raise ValueError(f"invalid library status: {status!r}")
        if not target_lang:
            raise ValueError("target_lang required")
        for _, origin in (materials or ()):
            if origin not in ("auto", "manual"):
                raise ValueError(f"invalid link_origin: {origin!r}")
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO library_entries "
                "  (user_id, work_id, title, cover_url, target_lang, status) "
                "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                user_id, work_id, title, cover_url, target_lang, status,
            )
            entry_id = int(row["id"])
            for material_id, origin in (materials or ()):
                await conn.execute(
                    "INSERT INTO library_materials "
                    "  (entry_id, material_id, user_id, link_origin) "
                    "VALUES ($1, $2, $3, $4) "
                    "ON CONFLICT (entry_id, material_id) DO NOTHING",
                    entry_id, material_id, user_id, origin,
                )
        return entry_id

    async def update_library_entry(
        self,
        entry_id: int, user_id: int,
        *,
        title:       str | None = None,
        status:      str | None = None,
        target_lang: str | None = None,
    ) -> None:
        sets: list[str] = []
        args: list = []
        if title is not None:
            args.append(title); sets.append(f"title=${len(args)}")
        if status is not None:
            if status not in ("reading", "plan", "done", "dropped"):
                raise ValueError(f"invalid library status: {status!r}")
            args.append(status); sets.append(f"status=${len(args)}")
        if target_lang is not None:
            tl = target_lang.strip()
            if not tl:
                raise ValueError("target_lang cannot be blank")
            args.append(tl); sets.append(f"target_lang=${len(args)}")
        if not sets:
            return
        args.extend([entry_id, user_id])
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE library_entries SET {', '.join(sets)} "
                f"WHERE id=${len(args)-1} AND user_id=${len(args)}",
                *args,
            )

    async def delete_library_entry(
        self, entry_id: int, user_id: int,
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM library_entries WHERE id=$1 AND user_id=$2",
                entry_id, user_id,
            )
        return result.startswith("DELETE ") and not result.endswith("0")

    async def link_material_to_entry(
        self,
        *,
        entry_id:    int,
        material_id: int,
        link_origin: str,
        voter_id:    int,
    ) -> None:
        """Link the material + cast +1 votes on every (existing, new)
        pair already in the entry. Multi-link → multiple vote rows."""
        if link_origin not in ("auto", "manual"):
            raise ValueError(f"invalid link_origin: {link_origin!r}")
        async with self._pool.acquire() as conn, conn.transaction():
            owner_row = await conn.fetchrow(
                "SELECT user_id FROM library_entries WHERE id=$1", entry_id,
            )
            if owner_row is None:
                raise ValueError(f"entry {entry_id} not found")
            entry_owner = owner_row["user_id"]
            existing = await conn.fetch(
                "SELECT material_id FROM library_materials WHERE entry_id=$1",
                entry_id,
            )
            await conn.execute(
                "INSERT INTO library_materials "
                "  (entry_id, material_id, user_id, link_origin) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (entry_id, material_id) DO NOTHING",
                entry_id, material_id, entry_owner, link_origin,
            )
            for r in existing:
                other = r["material_id"]
                if other == material_id:
                    continue
                a, b = _canonical_pair(other, material_id)
                await conn.execute(
                    "INSERT INTO material_link_votes "
                    "  (material_a_id, material_b_id, voter_id, vote) "
                    "VALUES ($1, $2, $3, 1) "
                    "ON CONFLICT (material_a_id, material_b_id, voter_id) "
                    "  DO UPDATE SET vote=1, voted_at=NOW()",
                    a, b, voter_id,
                )

    async def unlink_material_from_entry(
        self,
        *,
        entry_id:    int,
        material_id: int,
        voter_id:    int,
    ) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            siblings = await conn.fetch(
                "SELECT material_id FROM library_materials "
                "WHERE entry_id=$1 AND material_id != $2",
                entry_id, material_id,
            )
            await conn.execute(
                "DELETE FROM library_materials "
                "WHERE entry_id=$1 AND material_id=$2",
                entry_id, material_id,
            )
            for r in siblings:
                a, b = _canonical_pair(r["material_id"], material_id)
                await conn.execute(
                    "DELETE FROM material_link_votes "
                    "WHERE material_a_id=$1 AND material_b_id=$2 AND voter_id=$3",
                    a, b, voter_id,
                )



    # ── Community recent feed (cross-user, no guild scope) ───────

    async def list_recent_community(
        self,
        *,
        viewer_id: int,
        limit:     int = 60,
        before:    str | None = None,
    ) -> list[dict]:
        """Recent translations across the community, deduped per Work
        (latest chapter per manga wins). `viewer_id` is accepted for
        API symmetry but no longer gates visibility — schema 19 made
        every non-takedown translation public.

        Cursor pagination via ISO `before` filters on the per-Work
        latest `created_at` so a manga doesn't reappear on later pages
        once its newest chapter is consumed.
        """
        del viewer_id  # reserved for future personalisation
        cond_before = ""
        args: list = [limit]
        if before is not None:
            args.append(before)
            cond_before = (
                f"AND latest_created_at < ${len(args)}::timestamptz"
            )
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH visible AS (
                    SELECT
                        t.id              AS translation_id,
                        c.id              AS chapter_id,
                        wc.number_norm    AS chapter_number,
                        c.label           AS chapter_label,
                        c.position        AS chapter_position,
                        m.work_id         AS work_id,
                        m.id              AS material_id,
                        m.title           AS material_title,
                        m.cover_url       AS material_cover,
                        t.target_lang,
                        t.owner_id        AS creator_id,
                        u.display_name    AS creator_name,
                        t.created_at,
                        COUNT(*) OVER (PARTITION BY m.work_id) AS chapters_in_feed,
                        MAX(t.created_at) OVER (PARTITION BY m.work_id) AS latest_created_at
                    FROM translations t
                    JOIN translation_drafts d ON d.id = t.draft_id
                    JOIN chapters  c ON c.id = d.chapter_id
                    JOIN work_chapters wc ON wc.id = c.work_chapter_id
                    JOIN materials m ON m.id = c.material_id
                    LEFT JOIN users u ON u.id = t.owner_id
                    WHERE t.takedown_at IS NULL
                      AND t.shared = TRUE
                )
                SELECT DISTINCT ON (work_id)
                    translation_id, chapter_id, chapter_number, chapter_label,
                    work_id, material_id, material_title, material_cover,
                    target_lang, creator_id, creator_name,
                    to_char(
                        created_at AT TIME ZONE 'UTC',
                        '{_ISO_FMT}'
                    ) AS created_at,
                    chapters_in_feed
                FROM visible
                WHERE TRUE
                {cond_before}
                ORDER BY work_id, created_at DESC
                LIMIT $1
                """,
                *args,
            )
        out = [dict(r) for r in rows]
        out.sort(key=lambda r: r["created_at"] or "", reverse=True)
        return out

    # ── Reading history (per-user, system-recorded) ──────────────

    async def record_reading(
        self, *,
        user_id:          int,
        work_chapter_id:  int,
        last_material_id: int | None,
        translation_id:   int | None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO reading_history "
                "  (user_id, work_chapter_id, last_material_id, "
                "   translation_id, last_read_at) "
                "VALUES ($1, $2, $3, $4, NOW()) "
                "ON CONFLICT (user_id, work_chapter_id) DO UPDATE SET "
                "  last_material_id = COALESCE(EXCLUDED.last_material_id, "
                "                              reading_history.last_material_id), "
                "  translation_id   = COALESCE(EXCLUDED.translation_id, "
                "                              reading_history.translation_id), "
                "  last_read_at     = NOW()",
                user_id, work_chapter_id, last_material_id, translation_id,
            )

    async def list_recent_reads(
        self, *, user_id: int, limit: int = 30,
    ) -> list[dict]:
        """Recent unique Works the user has read, newest first. Dedupes
        per Work (so reading three chapters of one manga collapses to
        one row). Surfaces the most-recent (material, chapter) the
        user actually opened so the row can deep-link back to the
        same source.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT DISTINCT ON (m.work_id)
                    m.work_id         AS work_id,
                    m.id              AS material_id,
                    m.title           AS material_title,
                    m.cover_url       AS material_cover,
                    wc.id             AS work_chapter_id,
                    wc.number_norm    AS chapter_number,
                    wc.label          AS chapter_label,
                    rh.translation_id,
                    to_char(
                        rh.last_read_at AT TIME ZONE 'UTC',
                        '{_ISO_FMT}'
                    ) AS last_read_at
                FROM reading_history rh
                JOIN work_chapters wc ON wc.id = rh.work_chapter_id
                JOIN materials m ON m.id = rh.last_material_id
                WHERE rh.user_id = $1
                  AND rh.last_material_id IS NOT NULL
                ORDER BY m.work_id, rh.last_read_at DESC
                LIMIT $2
                """,
                user_id, limit,
            )
        out = [dict(r) for r in rows]
        out.sort(key=lambda r: r["last_read_at"] or "", reverse=True)
        return out

    # ── Glossary ─────────────────────────────────────────────────

    async def list_user_glossary(
        self,
        user_id: int,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ) -> list[dict]:
        clauses = ["owner_id=$1"]
        args: list = [user_id]
        if source_lang is not None:
            args.append(source_lang); clauses.append(f"source_lang=${len(args)}")
        if target_lang is not None:
            args.append(target_lang); clauses.append(f"target_lang=${len(args)}")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, source_lang, target_lang, source_term, "
                "       target_term, notes "
                f"FROM user_glossary WHERE {' AND '.join(clauses)} "
                "ORDER BY source_term",
                *args,
            )
        return [dict(r) for r in rows]

    async def upsert_user_glossary_term(
        self,
        user_id: int,
        source_lang: str, target_lang: str,
        source_term: str, target_term: str,
        notes: str | None = None,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO user_glossary "
                "  (owner_id, source_lang, target_lang, source_term, "
                "   target_term, notes) "
                "VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (owner_id, source_lang, target_lang, source_term) "
                "  DO UPDATE SET target_term=EXCLUDED.target_term, "
                "                notes=EXCLUDED.notes "
                "RETURNING id",
                user_id, source_lang, target_lang,
                source_term, target_term, notes,
            )
        return row["id"]

    async def delete_user_glossary_term(
        self, user_id: int, term_id: int,
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_glossary WHERE owner_id=$1 AND id=$2",
                user_id, term_id,
            )
        return result.startswith("DELETE ") and not result.endswith("0")

    async def compute_glossary_fingerprint(
        self,
        *,
        user_id: int,
        source_lang: str,
        target_lang: str,
        material_id: int | None,
    ) -> str:
        """Build the effective glossary signature for cache keying.

        Composition (in order; user entries override community):
            community_glossary WHERE material_id IS NULL  (global)
            community_glossary WHERE material_id = $X     (material-scoped)
            user_glossary                                  (overrides)

        SHA256 of the deterministic 'src=tgt|src=tgt|...' string,
        truncated to 16 hex chars. Two users with identical effective
        glossary collide → cache hit.
        """
        import hashlib
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_term, target_term FROM (
                    SELECT source_term, target_term, 1 AS prio
                    FROM community_glossary
                    WHERE source_lang=$1 AND target_lang=$2 AND material_id IS NULL
                    UNION ALL
                    SELECT source_term, target_term, 2 AS prio
                    FROM community_glossary
                    WHERE source_lang=$1 AND target_lang=$2 AND material_id=$3
                    UNION ALL
                    SELECT source_term, target_term, 3 AS prio
                    FROM user_glossary
                    WHERE source_lang=$1 AND target_lang=$2 AND owner_id=$4
                ) g
                ORDER BY source_term, prio DESC
                """,
                source_lang, target_lang, material_id, user_id,
            )
        # Take last entry per source_term so highest-priority override wins.
        merged: dict[str, str] = {}
        for r in rows:
            merged[r["source_term"]] = r["target_term"]
        sig = "|".join(f"{k}={v}" for k, v in sorted(merged.items()))
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]

    # ── Quota ────────────────────────────────────────────────────

    async def record_chapter_consume(
        self,
        *,
        user_id: int,
        translation_id: int,
        kind: str,
    ) -> None:
        if kind not in ("draft_create", "render_create"):
            raise ValueError(f"invalid consume kind: {kind!r}")
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chapter_consumes "
                "  (user_id, translation_id, kind) "
                "VALUES ($1, $2, $3)",
                user_id, translation_id, kind,
            )

    async def count_user_consumes_since(
        self, user_id: int, seconds: int,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) AS n FROM chapter_consumes "
                "WHERE user_id = $1 "
                "  AND created_at >= NOW() - make_interval(secs => $2)",
                user_id, seconds,
            )
        return int(row["n"]) if row else 0

    # ── Tasks queue ──────────────────────────────────────────────

    async def enqueue_task(
        self,
        *,
        target_kind: str,
        target_id:   int,
        stage:       str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO tasks (target_kind, target_id, stage) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (target_kind, target_id, stage) DO NOTHING",
                target_kind, target_id, stage,
            )

    async def claim_task(
        self,
        stage:     str,
        worker_id: str,
    ) -> tuple[str, int] | None:
        """Atomically claim one pending task. Returns (target_kind,
        target_id) or None.

        FOR UPDATE SKIP LOCKED gives us a single-statement claim with
        no race recheck. Tasks past MAX_TASK_ATTEMPTS are dead-lettered
        and never re-claimed. Stale claims older than STALE_CLAIM_SECONDS
        become re-claimable. Paused stages (`stage_pause`) are skipped
        entirely — workers will not touch any task under a stage that
        the operator has marked as broken.
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
                      AND  NOT EXISTS (
                          SELECT 1 FROM stage_pause sp WHERE sp.stage = $2
                      )
                    ORDER  BY target_id
                    FOR UPDATE SKIP LOCKED
                    LIMIT  1
                )
                RETURNING target_kind, target_id
                """,
                worker_id, stage, MAX_TASK_ATTEMPTS,
            )
        if row is None:
            return None
        return (row["target_kind"], row["target_id"])

    async def complete_task(
        self,
        target_kind: str, target_id: int,
        stage: str,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM tasks "
                "WHERE target_kind=$1 AND target_id=$2 AND stage=$3",
                target_kind, target_id, stage,
            )

    async def advance_task(
        self,
        target_kind: str, target_id: int,
        completed_stage: str, next_stage: str,
        *,
        next_target_kind: str | None = None,
        next_target_id:   int | None = None,
    ) -> None:
        """Complete current stage + enqueue next in one txn.

        The next stage may target a different kind (e.g. chapter scan
        finishes → enqueue draft translate). Caller passes
        next_target_* when the target shifts.
        """
        nk = next_target_kind if next_target_kind is not None else target_kind
        ni = next_target_id   if next_target_id   is not None else target_id
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM tasks "
                "WHERE target_kind=$1 AND target_id=$2 AND stage=$3",
                target_kind, target_id, completed_stage,
            )
            await conn.execute(
                "INSERT INTO tasks (target_kind, target_id, stage) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (target_kind, target_id, stage) DO NOTHING",
                nk, ni, next_stage,
            )

    async def fail_task(
        self,
        target_kind: str, target_id: int,
        stage: str, error: str,
    ) -> None:
        """Mark a task failed terminally — bump attempts to the cap so
        it's never re-claimed. Operator must redo or delete."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET claimed_by=NULL, claimed_at=NULL, "
                "  attempts=$1, last_error=$2 "
                "WHERE target_kind=$3 AND target_id=$4 AND stage=$5",
                MAX_TASK_ATTEMPTS, error, target_kind, target_id, stage,
            )

    async def release_task_for_transient(
        self,
        target_kind: str, target_id: int,
        stage: str, error: str,
    ) -> None:
        """Release the claim without touching `attempts`.

        Used for `UpstreamUnavailable` — the failure is expected to
        clear on its own. The task waits on the worker's backoff
        before being re-claimed; we never burn attempts, so an
        all-day outage doesn't dead-letter live drafts.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET claimed_by=NULL, claimed_at=NULL, "
                "  last_error=$1 "
                "WHERE target_kind=$2 AND target_id=$3 AND stage=$4",
                error, target_kind, target_id, stage,
            )

    async def pause_stage(
        self, stage: str, *, reason: str, paused_by: str | None = None,
    ) -> bool:
        """Mark a whole stage as blocked pending operator action.

        Idempotent: re-pausing keeps the first `reason` so post-mortem
        sees what actually broke. Returns True if this call inserted
        the row, False if the stage was already paused.

        While a stage is paused, `claim_task` will not return any of
        its tasks. Existing claims (if any) drain through their
        current dispatch and are released on completion / failure.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "INSERT INTO stage_pause (stage, reason, paused_by) "
                "VALUES ($1, $2, $3) "
                "ON CONFLICT (stage) DO NOTHING",
                stage, reason, paused_by,
            )
        return result.endswith(" 1")

    async def resume_stage(self, stage: str) -> bool:
        """Lift a pause. Workers wake on NOTIFY (DELETE trigger) and
        start claiming again. Returns True if the stage was paused,
        False if it wasn't (idempotent resume).
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM stage_pause WHERE stage=$1",
                stage,
            )
        return result.endswith(" 1")

    async def list_paused_stages(self) -> list[dict]:
        """Snapshot of all paused stages for the ops CLI / dashboard."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT stage, reason, paused_at, paused_by "
                "FROM stage_pause ORDER BY paused_at"
            )
        return [
            {
                "stage":     r["stage"],
                "reason":    r["reason"],
                "paused_at": r["paused_at"],
                "paused_by": r["paused_by"],
            }
            for r in rows
        ]

    async def release_claims_by_prefix(self, prefix: str) -> int:
        """Release every claim whose `claimed_by` starts with `prefix`.

        Called on worker startup with the local hostname to clear ghost
        claims left by a previous PID that died without graceful exit.
        Returns the count released. Does not touch `attempts` — the
        task was interrupted externally, not by stage-code exception.
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE tasks SET claimed_by=NULL, claimed_at=NULL "
                "WHERE claimed_by LIKE $1",
                f"{prefix}%",
            )
        try:
            return int(result.rsplit(" ", 1)[-1])
        except ValueError:
            return 0

    async def queue_stats(self) -> dict:
        """Snapshot for the header indicator + ops dashboard.

        `pending` counts only tasks that are actually re-claimable —
        dead-lettered rows (`attempts >= MAX_TASK_ATTEMPTS`) and rows
        under a paused stage are excluded so the header doesn't show
        a misleading "N chờ" while everything is actually stuck.
        Those are surfaced separately as `blocked` (paused stage) and
        `failed` (dead-letter) so the user sees the real state.
        """
        async with self._pool.acquire() as conn:
            stage_rows = await conn.fetch(
                f"""
                SELECT
                    t.stage,
                    SUM(CASE
                        WHEN t.claimed_by IS NULL
                         AND t.attempts < {MAX_TASK_ATTEMPTS}
                         AND NOT EXISTS (
                             SELECT 1 FROM stage_pause sp WHERE sp.stage = t.stage
                         )
                        THEN 1 ELSE 0 END
                    ) AS pending,
                    SUM(CASE
                        WHEN t.claimed_by IS NOT NULL
                         AND t.claimed_at >= NOW() - INTERVAL '10 minutes'
                        THEN 1 ELSE 0 END
                    ) AS running,
                    SUM(CASE
                        WHEN t.claimed_by IS NOT NULL
                         AND t.claimed_at <  NOW() - INTERVAL '10 minutes'
                        THEN 1 ELSE 0 END
                    ) AS stale,
                    SUM(CASE
                        WHEN t.claimed_by IS NULL
                         AND EXISTS (
                             SELECT 1 FROM stage_pause sp WHERE sp.stage = t.stage
                         )
                        THEN 1 ELSE 0 END
                    ) AS blocked,
                    SUM(CASE
                        WHEN t.claimed_by IS NULL
                         AND t.attempts >= {MAX_TASK_ATTEMPTS}
                        THEN 1 ELSE 0 END
                    ) AS failed
                FROM tasks t
                GROUP BY t.stage
                """
            )
            active_rows = await conn.fetch(
                "SELECT DISTINCT claimed_by FROM tasks "
                "WHERE claimed_by IS NOT NULL "
                "  AND claimed_at >= NOW() - INTERVAL '10 minutes'"
            )
            paused_rows = await conn.fetch(
                "SELECT stage FROM stage_pause"
            )
        stages: dict[str, dict[str, int]] = {
            r["stage"]: {
                "pending": int(r["pending"] or 0),
                "running": int(r["running"] or 0),
                "stale":   int(r["stale"]   or 0),
                "blocked": int(r["blocked"] or 0),
                "failed":  int(r["failed"]  or 0),
            }
            for r in stage_rows
        }
        active = [r["claimed_by"] for r in active_rows]
        paused = [r["stage"] for r in paused_rows]
        return {
            "stages":         stages,
            "active_workers": active,
            "paused_stages":  paused,
        }

    # ── Chapter inbox (deferred prepare handle) ───────────────────

    async def set_inbox_handle(self, handle: "InboxHandle") -> None:
        # `jsonb` codec (see `_init_connection`) handles encoding the
        # Python list to JSON text — pass the structure as-is.
        parts_payload = [
            {"number": p.number, "etag": p.etag} for p in handle.parts
        ]
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO material_inbox "
                "  (chapter_id, tmp_id, upload_id, parts, title) "
                "VALUES ($1, $2, $3, $4::jsonb, $5) "
                "ON CONFLICT (chapter_id) DO UPDATE SET "
                "  tmp_id=EXCLUDED.tmp_id, upload_id=EXCLUDED.upload_id, "
                "  parts=EXCLUDED.parts, title=EXCLUDED.title",
                handle.chapter_id, handle.tmp_id, handle.upload_id,
                parts_payload, handle.title,
            )

    async def get_inbox_handle(
        self, chapter_id: int,
    ) -> "InboxHandle | None":
        from typoon.adapters.inbox import CompletedPart, InboxHandle
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT tmp_id, upload_id, parts, title "
                "FROM material_inbox WHERE chapter_id=$1",
                chapter_id,
            )
        if row is None:
            return None
        raw_parts = row["parts"]
        # Backward-compat: older rows from a buggy writer (pre-fix)
        # stored `parts` as a JSON-encoded string nested inside the
        # jsonb column. The codec decodes once to `str`; decode again
        # so worker prepare doesn't crash on those legacy rows.
        if isinstance(raw_parts, str):
            raw_parts = json.loads(raw_parts)
        return InboxHandle(
            chapter_id=chapter_id,
            tmp_id=row["tmp_id"],
            upload_id=row["upload_id"],
            parts=tuple(
                CompletedPart(number=int(p["number"]), etag=str(p["etag"]))
                for p in raw_parts
            ),
            title=row["title"],
        )

    async def clear_inbox_handle(self, chapter_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM material_inbox WHERE chapter_id=$1",
                chapter_id,
            )

    # ── Translator memory ────────────────────────────────────────
    #
    # Per (user, material, target_lang) knowledge state. Reads merge the
    # row with its sliding-window briefs at the route layer; this class
    # only exposes raw row access + brief append/list.

    async def get_translator_memory(
        self,
        *,
        user_id:     int,
        material_id: int,
        target_lang: str,
    ) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_TS_TRANSLATOR_MEMORY} FROM translator_memory "
                "WHERE user_id=$1 AND material_id=$2 AND target_lang=$3",
                user_id, material_id, target_lang,
            )
        return _row_dict(row)

    async def upsert_translator_memory(
        self,
        *,
        user_id:      int,
        material_id:  int,
        source_lang:  str,
        target_lang:  str,
        characters:   list | None = None,
        world:        dict | None = None,
        style:        dict | None = None,
        glossary:     list | None = None,
        style_refs:   list | None = None,
    ) -> dict:
        """Create or update memory cards. None values keep the existing
        column; an empty list/dict explicitly clears the field. Callers
        that want to *append* to characters/glossary should read,
        mutate, write — the row is small enough that load-modify-store
        is fine."""
        async with self._pool.acquire() as conn, conn.transaction():
            existing = await conn.fetchrow(
                "SELECT id FROM translator_memory "
                "WHERE user_id=$1 AND material_id=$2 AND target_lang=$3",
                user_id, material_id, target_lang,
            )
            if existing is None:
                await conn.execute(
                    "INSERT INTO translator_memory ("
                    "  user_id, material_id, source_lang, target_lang,"
                    "  characters, world, style, glossary, style_refs"
                    ") VALUES ("
                    "  $1, $2, $3, $4,"
                    "  $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb"
                    ")",
                    user_id, material_id, source_lang, target_lang,
                    characters if characters is not None else [],
                    world      if world      is not None else {},
                    style      if style      is not None else {},
                    glossary   if glossary   is not None else [],
                    style_refs if style_refs is not None else [],
                )
            else:
                sets: list[str] = []
                args: list = []
                # source_lang is intentionally immutable post-insert —
                # changing src mid-stream would invalidate every brief.
                if characters is not None:
                    args.append(characters)
                    sets.append(f"characters=${len(args)}::jsonb")
                if world is not None:
                    args.append(world)
                    sets.append(f"world=${len(args)}::jsonb")
                if style is not None:
                    args.append(style)
                    sets.append(f"style=${len(args)}::jsonb")
                if glossary is not None:
                    args.append(glossary)
                    sets.append(f"glossary=${len(args)}::jsonb")
                if style_refs is not None:
                    args.append(style_refs)
                    sets.append(f"style_refs=${len(args)}::jsonb")
                if sets:
                    args.extend([user_id, material_id, target_lang])
                    await conn.execute(
                        f"UPDATE translator_memory SET {', '.join(sets)} "
                        f"WHERE user_id=${len(args)-2} "
                        f"AND material_id=${len(args)-1} "
                        f"AND target_lang=${len(args)}",
                        *args,
                    )
            row = await conn.fetchrow(
                f"SELECT {_TS_TRANSLATOR_MEMORY} FROM translator_memory "
                "WHERE user_id=$1 AND material_id=$2 AND target_lang=$3",
                user_id, material_id, target_lang,
            )
        return _row_dict(row)  # type: ignore[return-value]

    async def append_memory_brief(
        self,
        *,
        memory_id:  int,
        chapter_id: int,
        brief_json: dict,
        summary:    str | None,
    ) -> None:
        """Insert-or-replace the brief for this chapter. Each (memory,
        chapter) collapses to one row — repeated translates of the same
        chapter overwrite the prior brief so the sliding window stays
        coherent."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO translator_memory_briefs ("
                "  memory_id, chapter_id, brief_json, summary"
                ") VALUES ($1, $2, $3::jsonb, $4) "
                "ON CONFLICT (memory_id, chapter_id) DO UPDATE SET "
                "  brief_json = EXCLUDED.brief_json, "
                "  summary    = EXCLUDED.summary, "
                "  updated_at = NOW()",
                memory_id, chapter_id,
                brief_json, summary,
            )
            # Advance last_chapter_id when this chapter sits past the
            # current high-water mark (compare by chapter.position).
            await conn.execute(
                """
                UPDATE translator_memory tm
                SET    last_chapter_id = c.id
                FROM   chapters c, chapters cur
                WHERE  tm.id = $1
                  AND  c.id  = $2
                  AND  cur.id = COALESCE(tm.last_chapter_id, c.id)
                  AND  c.position >= cur.position
                """,
                memory_id, chapter_id,
            )

    async def list_recent_memory_briefs(
        self,
        *,
        memory_id:    int,
        before_chapter_id: int | None = None,
        limit:        int = 5,
    ) -> list[dict]:
        """Most-recent briefs strictly before `before_chapter_id`
        (when given), keyed by chapter.position. With limit=5 this is
        the sliding-window the context agent injects when translating
        the next chapter."""
        sql = (
            "SELECT b.chapter_id, c.position, wc.number_norm AS number, c.label, "
            "       b.brief_json, b.summary, "
            f"      {_ts('b.created_at', 'created_at')}, "
            f"      {_ts('b.updated_at', 'updated_at')} "
            "FROM   translator_memory_briefs b "
            "JOIN   chapters c ON c.id = b.chapter_id "
            "JOIN   work_chapters wc ON wc.id = c.work_chapter_id "
            "WHERE  b.memory_id = $1"
        )
        args: list = [memory_id]
        if before_chapter_id is not None:
            args.append(before_chapter_id)
            sql += (
                f" AND c.position < (SELECT position FROM chapters "
                f"                   WHERE id=${len(args)})"
            )
        args.append(limit)
        sql += f" ORDER BY c.position DESC LIMIT ${len(args)}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]

    async def delete_translator_memory(
        self,
        *,
        user_id:     int,
        material_id: int,
        target_lang: str,
    ) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM translator_memory "
                "WHERE user_id=$1 AND material_id=$2 AND target_lang=$3",
                user_id, material_id, target_lang,
            )
        return result.startswith("DELETE ") and not result.endswith("0")

    # ── Reports + moderation ─────────────────────────────────────

    async def submit_report(
        self,
        *,
        reporter_id:    int | None,
        reporter_label: str,
        target_kind:    str,
        target_id:      int,
        kind:           str,
        reason:         str,
    ) -> int:
        if target_kind not in ("material", "chapter", "draft", "translation"):
            raise ValueError(f"invalid target_kind: {target_kind!r}")
        if kind not in ("dmca", "abuse", "quality", "other"):
            raise ValueError(f"invalid report kind: {kind!r}")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO reports ("
                "  reporter_id, reporter_label, "
                "  target_kind, target_id, "
                "  kind, reason"
                ") VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                reporter_id, reporter_label,
                target_kind, target_id,
                kind, reason,
            )
        return row["id"]

    async def get_report(self, report_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, reporter_id, reporter_label, "
                "       target_kind, target_id, "
                "       kind, reason, status, "
                f"       {_ts('created_at')}, "
                f"       {_ts('resolved_at')}, resolved_by "
                "FROM reports WHERE id=$1",
                report_id,
            )
        return _row_dict(row)

    async def list_reports(
        self,
        *,
        status: str | None = None,
        limit:  int = 100,
    ) -> list[dict]:
        clauses: list[str] = []
        args: list = []
        if status is not None:
            args.append(status); clauses.append(f"status=${len(args)}")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        args.append(limit)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, reporter_id, reporter_label, "
                "       target_kind, target_id, "
                "       kind, reason, status, "
                f"       {_ts('created_at')}, "
                f"       {_ts('resolved_at')}, resolved_by "
                f"FROM reports {where} "
                f"ORDER BY created_at DESC LIMIT ${len(args)}",
                *args,
            )
        return [dict(r) for r in rows]

    async def update_report_status(
        self,
        report_id:   int,
        *,
        status:      str,
        resolver_id: int | None,
    ) -> bool:
        if status not in ("open", "reviewing", "resolved", "dismissed"):
            raise ValueError(f"invalid report status: {status!r}")
        terminal = status in ("resolved", "dismissed")
        async with self._pool.acquire() as conn:
            if terminal:
                result = await conn.execute(
                    "UPDATE reports "
                    "SET status=$2, resolved_at=NOW(), resolved_by=$3 "
                    "WHERE id=$1",
                    report_id, status, resolver_id,
                )
            else:
                result = await conn.execute(
                    "UPDATE reports "
                    "SET status=$2, resolved_at=NULL, resolved_by=NULL "
                    "WHERE id=$1",
                    report_id, status,
                )
        return result.startswith("UPDATE ") and not result.endswith("0")

    async def apply_moderation_action(
        self,
        *,
        report_id:   int | None,
        target_kind: str,
        target_id:   int,
        action:      str,
        reason:      str,
        actor_id:    int | None,
    ) -> int:
        if target_kind not in ("material", "chapter", "draft", "translation"):
            raise ValueError(f"invalid target_kind: {target_kind!r}")
        if action not in ("takedown", "restore", "delete"):
            raise ValueError(f"invalid action: {action!r}")
        # Map (action, target_kind) → SQL effect.
        soft_targets = {
            "draft":       "translation_drafts",
            "translation": "translations",
        }
        hard_targets = {
            "material":    "materials",
            "chapter":     "chapters",
        }
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO moderation_actions ("
                "  report_id, target_kind, target_id, "
                "  action, reason, actor_id"
                ") VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                report_id, target_kind, target_id,
                action, reason, actor_id,
            )
            if action == "takedown":
                table = soft_targets.get(target_kind)
                if table is None:
                    raise ValueError(
                        f"takedown only applies to draft/translation; "
                        f"got {target_kind!r}"
                    )
                await conn.execute(
                    f"UPDATE {table} "
                    "SET takedown_at=NOW(), takedown_reason=$2 "
                    "WHERE id=$1",
                    target_id, reason,
                )
            elif action == "restore":
                table = soft_targets.get(target_kind)
                if table is None:
                    raise ValueError(
                        f"restore only applies to draft/translation; "
                        f"got {target_kind!r}"
                    )
                await conn.execute(
                    f"UPDATE {table} "
                    "SET takedown_at=NULL, takedown_reason=NULL "
                    "WHERE id=$1",
                    target_id,
                )
            else:  # delete — hard delete on material/chapter only
                table = hard_targets.get(target_kind)
                if table is None:
                    raise ValueError(
                        f"delete only applies to material/chapter; "
                        f"got {target_kind!r}"
                    )
                await conn.execute(
                    f"DELETE FROM {table} WHERE id=$1", target_id,
                )
        return row["id"]

    async def list_moderation_actions_for_target(
        self,
        *,
        target_kind: str,
        target_id:   int,
        limit:       int = 50,
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, report_id, target_kind, target_id, "
                "       action, reason, actor_id, "
                f"       {_ts('created_at')} "
                "FROM moderation_actions "
                "WHERE target_kind=$1 AND target_id=$2 "
                "ORDER BY created_at DESC LIMIT $3",
                target_kind, target_id, limit,
            )
        return [dict(r) for r in rows]
