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
SCHEMA_VERSION = "15"  # v5 material+translation architecture (RFC v5)

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


def _ts(col: str) -> str:
    return f"to_char(({col}) AT TIME ZONE 'UTC', '{_ISO_FMT}') AS {col}"


_TS_USERS = (
    "id, display_name, avatar_url, email, "
    f"{_ts('created_at')}, {_ts('last_login_at')}"
)
_TS_MATERIAL = (
    "id, imported_by, origin, source, upstream_ref, title, cover_url, "
    "description, author, status, languages, title_native, title_alt, "
    "cross_refs, nsfw, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_CHAPTER = (
    "id, material_id, position, number, label, upstream_url, "
    "pages_origin, prepared_hash, prepared_backend, prepared_locator, "
    "masks_backend, masks_locator, page_count, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_DRAFT = (
    "id, chapter_id, source_lang, target_lang, glossary_fp, llm_model, "
    "created_by, visibility, scope_guild_id, state, error_message, "
    "progress_stage, progress_index, progress_total, "
    f"{_ts('takedown_at')}, takedown_reason, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_TRANSLATION = (
    "id, chapter_id, owner_id, target_lang, draft_id, "
    "archive_backend, archive_locator, "
    f"{_ts('rendered_at')}, "
    "in_feed, feed_guild_id, "
    f"{_ts('takedown_at')}, takedown_reason, "
    f"{_ts('created_at')}, {_ts('updated_at')}"
)
_TS_LIBRARY_ENTRY = (
    "id, user_id, title, cover_url, primary_material_id, "
    f"bookmarked, {_ts('bookmarked_at')}, "
    f"{_ts('last_read_at')}, last_chapter_ref, "
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
    conn: asyncpg.Connection, material_id: int, number: str,
) -> int:
    """Compute the `position` for a new chapter being inserted.

    Caller must hold the material's advisory xact lock.

    Strategy:
      - "Extra"/"Oneshot"/anything not parseable as a float → append at
        max(position) + INITIAL_GAP.
      - Numeric `number` → place between the last sibling whose number
        parses to ≤ target and the first sibling > target. Equality
        goes to `prev` so a later upload of the same `number` lands
        AFTER the existing one (first-come stays first).
      - When the chosen gap is below REBALANCE_MIN_GAP, redistribute
        the whole material to INITIAL_GAP spacing and retry once.
    """
    target = _try_float(number)
    rows = await conn.fetch(
        "SELECT position, number FROM chapters "
        "WHERE material_id=$1 ORDER BY position",
        material_id,
    )
    if not rows:
        return INITIAL_GAP

    if target is None:
        return rows[-1]["position"] + INITIAL_GAP

    prev_pos: int | None = None
    next_pos: int | None = None
    for r in rows:
        n = _try_float(r["number"])
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
        return await _resolve_chapter_position(conn, material_id, number)
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
    async def open(dsn: str) -> "PostgresStore":
        if not dsn or not dsn.startswith(("postgresql://", "postgres://")):
            raise RuntimeError(
                f"DATABASE_URL must be a postgresql:// DSN, got: {dsn!r}"
            )
        # statement_cache_size=0 disables asyncpg's per-connection
        # prepared-statement cache. Sporadic "_get_statement" failures
        # under concurrent first requests — the cache was a small win
        # not worth the flake. Re-enable later if profiling justifies.
        pool = await asyncpg.create_pool(
            dsn, min_size=2, max_size=10, statement_cache_size=0,
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
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT external_id FROM identities "
                "WHERE user_id=$1 AND provider=$2 LIMIT 1",
                user_id, provider,
            )
        return row["external_id"] if row else None

    # ── Guild memberships ─────────────────────────────────────────

    async def upsert_user_guilds(
        self, user_id: int, guilds: list[dict],
    ) -> None:
        """Replace cached guilds for this user. Items: `{id, name?, icon_url?}`.
        Replace strategy keeps user_guilds in sync with the current
        Discord state — if user leaves a guild, the row goes away on
        next login."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "DELETE FROM user_guilds WHERE user_id=$1", user_id,
            )
            if not guilds:
                return
            await conn.executemany(
                "INSERT INTO user_guilds (user_id, guild_id, guild_name, guild_icon) "
                "VALUES ($1, $2, $3, $4)",
                [
                    (user_id, g["id"], g.get("name"), g.get("icon_url"))
                    for g in guilds
                ],
            )

    async def get_user_guilds(self, user_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT guild_id AS id, guild_name AS name, "
                "       guild_icon AS icon_url "
                "FROM user_guilds WHERE user_id=$1 "
                "ORDER BY refreshed_at DESC",
                user_id,
            )
        return [dict(r) for r in rows]

    async def user_in_guild(self, user_id: int, guild_id: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM user_guilds WHERE user_id=$1 AND guild_id=$2",
                user_id, guild_id,
            )
        return row is not None

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
        rows untouched. A later background job may refresh metadata."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO materials ("
                "  imported_by, origin, source, upstream_ref, "
                "  title, cover_url, description, author, status, "
                "  languages, title_native, title_alt, cross_refs, nsfw"
                ") VALUES ($1, 'source', $2, $3, $4, $5, $6, $7, $8, "
                "          $9, $10, $11, $12::jsonb, $13) "
                "ON CONFLICT (source, upstream_ref) "
                "  WHERE source IS NOT NULL DO UPDATE "
                "  SET source = EXCLUDED.source "  # no-op, just to return id
                "RETURNING id",
                imported_by, source, upstream_ref, title, cover_url,
                description, author, status,
                list(languages or []), title_native, list(title_alt or []),
                json.dumps(cross_refs) if cross_refs else None, nsfw,
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
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO materials ("
                "  imported_by, origin, title, cover_url, description, "
                "  author, nsfw"
                ") VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
                imported_by, origin, title, cover_url, description, author, nsfw,
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

    # ── Chapter ───────────────────────────────────────────────────

    async def create_chapter(
        self,
        material_id:  int,
        number:       str,
        *,
        label:        str | None = None,
        upstream_url: str | None = None,
        pages_origin: str = "remote",
    ) -> int:
        if pages_origin not in ("remote", "local"):
            raise ValueError(
                f"pages_origin must be remote|local, got {pages_origin!r}"
            )
        async with self._pool.acquire() as conn, conn.transaction():
            # Advisory xact lock on material to serialize position
            # assignment within a single material's chapters.
            await conn.execute(
                "SELECT pg_advisory_xact_lock($1)", material_id,
            )
            pos = await _resolve_chapter_position(conn, material_id, number)
            row = await conn.fetchrow(
                "INSERT INTO chapters ("
                "  material_id, position, number, label, upstream_url, pages_origin"
                ") VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                material_id, pos, number, label, upstream_url, pages_origin,
            )
        return row["id"]

    async def get_chapter(self, chapter_id: int) -> dict | None:
        async with self._pool.acquire() as conn:
            return _row_dict(await conn.fetchrow(
                f"SELECT {_TS_CHAPTER} FROM chapters WHERE id=$1",
                chapter_id,
            ))

    async def list_chapters(self, material_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_TS_CHAPTER} FROM chapters "
                "WHERE material_id=$1 ORDER BY position",
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
        where each bubble has page_index, bubble_idx, polygon, fit_box,
        erase_box, text_box. Atomic replace per chapter."""
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
                            (chapter_id, b["page_index"], b["bubble_idx"],
                             json.dumps(b["polygon"]),
                             json.dumps(b["fit_box"]),
                             json.dumps(b["erase_box"]),
                             json.dumps(b["text_box"]))
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
            for k in ("polygon", "fit_box", "erase_box", "text_box"):
                if isinstance(d[k], str):
                    d[k] = json.loads(d[k])
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
        chapter_id:    int,
        source_lang:   str,
        target_lang:   str,
        glossary_fp:   str,
        viewer_id:     int,
        viewer_guilds: list[str],
    ) -> dict | None:
        """Cache lookup with visibility gate. Returns None if no draft
        matches the key, or if all matches are private/taken-down/
        out-of-scope for the viewer.

        Visibility rules:
          - 'guild':      scope_guild_id in viewer_guilds
          - 'all_guilds': creator shares ≥1 guild with viewer
          - 'private':    excluded by unique-cache index
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
                  AND  d.visibility != 'private'
                  AND  d.takedown_at IS NULL
                  AND  (
                      -- viewer is creator
                      d.created_by = $5
                      -- guild scope matches viewer's guilds
                      OR (d.visibility = 'guild'
                          AND d.scope_guild_id = ANY($6::text[]))
                      -- all_guilds: creator shares a guild with viewer
                      OR (d.visibility = 'all_guilds' AND EXISTS (
                          SELECT 1 FROM user_guilds ug
                          WHERE ug.user_id = d.created_by
                            AND ug.guild_id = ANY($6::text[])
                      ))
                  )
                ORDER BY d.state = 'done' DESC, d.created_at DESC
                LIMIT 1
                """,
                chapter_id, source_lang, target_lang, glossary_fp,
                viewer_id, viewer_guilds,
            )
        return _row_dict(row)

    async def create_draft(
        self,
        *,
        chapter_id:     int,
        source_lang:    str,
        target_lang:    str,
        glossary_fp:    str,
        llm_model:      str,
        created_by:     int,
        visibility:     str,
        scope_guild_id: str | None,
    ) -> int:
        if visibility not in ("private", "guild", "all_guilds"):
            raise ValueError(f"invalid visibility: {visibility!r}")
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO translation_drafts ("
                "  chapter_id, source_lang, target_lang, glossary_fp, "
                "  llm_model, created_by, visibility, scope_guild_id, state"
                ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending') "
                "RETURNING id",
                chapter_id, source_lang, target_lang, glossary_fp,
                llm_model, created_by, visibility, scope_guild_id,
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
                draft_id, json.dumps(brief, ensure_ascii=False),
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
        raw = row["brief_json"]
        return json.loads(raw) if isinstance(raw, str) else raw

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

    # ── Translations (Layer 3, per-user) ──────────────────────────

    async def get_or_create_translation(
        self,
        *,
        chapter_id:  int,
        owner_id:    int,
        target_lang: str,
        draft_id:    int | None,
    ) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO translations "
                "  (chapter_id, owner_id, target_lang, draft_id) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (chapter_id, owner_id, target_lang) DO UPDATE "
                "  SET draft_id=COALESCE(EXCLUDED.draft_id, translations.draft_id) "
                "RETURNING id",
                chapter_id, owner_id, target_lang, draft_id,
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
        chapter_ids:   list[int],
        viewer_id:     int,
        viewer_guilds: list[str],
    ) -> dict[int, list[dict]]:
        """Bulk overlay for chapter list views.

        Visibility filter mirrors `find_reusable_draft` but for whole
        translations: viewer sees their own + guild-shared via the
        draft pointer + drafts with all_guilds visibility whose creator
        shares a guild with the viewer.
        """
        if not chapter_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    t.id, t.chapter_id, t.owner_id, t.target_lang,
                    t.draft_id, t.in_feed, t.feed_guild_id,
                    COALESCE(d.state, 'done') AS state,
                    u.display_name AS creator_name,
                    (t.archive_locator IS NULL AND t.draft_id IS NOT NULL) AS uses_default_render
                FROM translations t
                LEFT JOIN translation_drafts d ON d.id = t.draft_id
                LEFT JOIN users u ON u.id = t.owner_id
                WHERE t.chapter_id = ANY($1::bigint[])
                  AND t.takedown_at IS NULL
                  AND (
                      t.owner_id = $2
                      OR d.visibility = 'guild' AND d.scope_guild_id = ANY($3::text[])
                      OR d.visibility = 'all_guilds' AND EXISTS (
                          SELECT 1 FROM user_guilds ug
                          WHERE ug.user_id = d.created_by
                            AND ug.guild_id = ANY($3::text[])
                      )
                  )
                ORDER BY t.chapter_id, t.created_at DESC
                """,
                chapter_ids, viewer_id, viewer_guilds,
            )
        out: dict[int, list[dict]] = {cid: [] for cid in chapter_ids}
        for r in rows:
            out[r["chapter_id"]].append(dict(r))
        return out

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

    async def update_translation_feed(
        self,
        translation_id: int,
        *,
        in_feed:        bool,
        feed_guild_id:  str | None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE translations "
                "SET in_feed=$1, feed_guild_id=$2 "
                "WHERE id=$3",
                in_feed, feed_guild_id, translation_id,
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

    async def list_library_entries(self, user_id: int) -> list[dict]:
        async with self._pool.acquire() as conn:
            entries = await conn.fetch(
                f"SELECT {_TS_LIBRARY_ENTRY} FROM library_entries "
                "WHERE user_id=$1 "
                "ORDER BY COALESCE(last_read_at, bookmarked_at, created_at) DESC",
                user_id,
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
        by_entry: dict[int, list[dict]] = {i: [] for i in ids}
        for l in links:
            d = dict(l)
            eid = d.pop("entry_id")
            by_entry.setdefault(eid, []).append(d)
        out: list[dict] = []
        for e in entries:
            d = dict(e)
            raw = d.get("last_chapter_ref")
            if isinstance(raw, str):
                d["last_chapter_ref"] = json.loads(raw)
            d["materials"] = by_entry.get(d["id"], [])
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
        d = dict(entry)
        raw = d.get("last_chapter_ref")
        if isinstance(raw, str):
            d["last_chapter_ref"] = json.loads(raw)
        d["materials"] = [dict(l) for l in links]
        return d

    async def create_library_entry(
        self,
        *,
        user_id:             int,
        title:               str,
        cover_url:           str | None,
        primary_material_id: int,
    ) -> int:
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO library_entries "
                "  (user_id, title, cover_url, primary_material_id) "
                "VALUES ($1, $2, $3, $4) RETURNING id",
                user_id, title, cover_url, primary_material_id,
            )
            entry_id = row["id"]
            await conn.execute(
                "INSERT INTO library_materials "
                "  (entry_id, material_id, user_id, link_origin) "
                "VALUES ($1, $2, $3, 'primary')",
                entry_id, primary_material_id, user_id,
            )
        return entry_id

    async def update_library_entry(
        self,
        entry_id: int, user_id: int,
        *,
        title:            str | None = None,
        bookmarked:       bool | None = None,
        last_read_at:     str | None = None,
        last_chapter_ref: dict | None = None,
    ) -> None:
        sets: list[str] = []
        args: list = []
        if title is not None:
            args.append(title); sets.append(f"title=${len(args)}")
        if bookmarked is not None:
            args.append(bookmarked); sets.append(f"bookmarked=${len(args)}")
            if bookmarked:
                sets.append("bookmarked_at=NOW()")
            else:
                sets.append("bookmarked_at=NULL")
        if last_read_at is not None:
            args.append(last_read_at)
            sets.append(f"last_read_at=${len(args)}::timestamptz")
        if last_chapter_ref is not None:
            args.append(json.dumps(last_chapter_ref))
            sets.append(f"last_chapter_ref=${len(args)}::jsonb")
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
        if link_origin not in ("primary", "auto", "manual"):
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

    async def find_library_suggestion(
        self,
        *,
        user_id:     int,
        material_id: int,
    ) -> dict | None:
        """Suggestion ranking per RFC §7.4.1. Returns a dict
        compatible with `LibrarySuggestionOut` or None."""
        async with self._pool.acquire() as conn:
            m = await conn.fetchrow(
                "SELECT id, title, title_native, cross_refs "
                "FROM materials WHERE id=$1",
                material_id,
            )
            if m is None:
                return None
            in_lib = await conn.fetchrow(
                "SELECT 1 FROM library_materials WHERE material_id=$1 "
                "  AND entry_id IN ("
                "    SELECT id FROM library_entries WHERE user_id=$2"
                "  ) LIMIT 1",
                material_id, user_id,
            )
            if in_lib is not None:
                return None  # already linked, nothing to suggest

            # Signal 1: cross_refs intersect
            if m["cross_refs"]:
                refs = m["cross_refs"]
                if isinstance(refs, str):
                    refs = json.loads(refs)
                if isinstance(refs, dict) and refs:
                    row = await conn.fetchrow(
                        "SELECT le.id AS entry_id, le.title AS entry_title "
                        "FROM library_entries le "
                        "JOIN library_materials lm ON lm.entry_id=le.id "
                        "JOIN materials m2 ON m2.id=lm.material_id "
                        "WHERE le.user_id=$1 "
                        "  AND m2.cross_refs IS NOT NULL "
                        "  AND m2.cross_refs @> $2::jsonb "
                        "LIMIT 1",
                        user_id, json.dumps(refs),
                    )
                    if row:
                        return {
                            "entry_id":    row["entry_id"],
                            "entry_title": row["entry_title"],
                            "confidence":  "high",
                            "signal":      "cross_refs",
                            "score":       None,
                        }

            # Signal 2/4: community vote score (high if ≥3, low if 1-2).
            # Reads live `material_link_votes` so a fresh vote (e.g. by
            # link_material_to_entry within the same session) is visible
            # without needing REFRESH MATERIALIZED VIEW.
            row = await conn.fetchrow(
                """
                SELECT le.id AS entry_id, le.title AS entry_title,
                       agg.score AS score
                FROM library_entries le
                JOIN library_materials lm ON lm.entry_id=le.id
                JOIN LATERAL (
                    SELECT SUM(vote)::INTEGER AS score
                    FROM material_link_votes
                    WHERE (material_a_id = lm.material_id AND material_b_id = $2)
                       OR (material_b_id = lm.material_id AND material_a_id = $2)
                ) agg ON agg.score >= 1
                WHERE le.user_id=$1
                ORDER BY agg.score DESC
                LIMIT 1
                """,
                user_id, material_id,
            )
            if row:
                s = int(row["score"])
                if s >= 3:
                    return {
                        "entry_id":    row["entry_id"],
                        "entry_title": row["entry_title"],
                        "confidence":  "high",
                        "signal":      "vote_high",
                        "score":       s,
                    }
                # Defer the low-score answer below in case signal 3 wins.

            # Signal 3: title_native case-fold match
            if m["title_native"]:
                row3 = await conn.fetchrow(
                    "SELECT le.id AS entry_id, le.title AS entry_title "
                    "FROM library_entries le "
                    "JOIN library_materials lm ON lm.entry_id=le.id "
                    "JOIN materials m2 ON m2.id=lm.material_id "
                    "WHERE le.user_id=$1 "
                    "  AND m2.id != $2 "
                    "  AND lower(m2.title_native) = lower($3) "
                    "LIMIT 1",
                    user_id, material_id, m["title_native"],
                )
                if row3:
                    return {
                        "entry_id":    row3["entry_id"],
                        "entry_title": row3["entry_title"],
                        "confidence":  "medium",
                        "signal":      "title_native",
                        "score":       None,
                    }

            # Signal 4: low-confidence vote (deferred from above)
            if row:
                s = int(row["score"])
                if 1 <= s <= 2:
                    return {
                        "entry_id":    row["entry_id"],
                        "entry_title": row["entry_title"],
                        "confidence":  "low",
                        "signal":      "vote_low",
                        "score":       s,
                    }
        return None

    async def reject_library_suggestion(
        self,
        *,
        voter_id:    int,
        material_id: int,
        candidate_material_id: int,
    ) -> None:
        """User said "không phải cùng manga" → -1 vote on the pair."""
        await self.cast_material_link_vote(
            voter_id, material_id, candidate_material_id, vote=-1,
        )

    # ── Feed (Hội Mê Truyện, guild-scoped) ───────────────────────

    async def list_feed_entries(
        self,
        *,
        guild_id:  str,
        viewer_id: int,
        limit:     int = 50,
        before:    str | None = None,
    ) -> list[dict]:
        """Translations with in_feed=TRUE scoped to the guild. Caller
        already enforced viewer is a member of guild_id.

        Two visibility paths feed the result:
          - feed_guild_id = guild_id (explicit publish to this guild)
          - feed_guild_id IS NULL  AND the translation's draft has
            visibility='all_guilds' AND the creator belongs to guild_id
            (publish to every guild creator is in).
        """
        cond_before = ""
        args: list = [guild_id, limit]
        if before is not None:
            args.append(before)
            cond_before = f"AND t.created_at < ${len(args)}::timestamptz"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    t.id              AS translation_id,
                    t.chapter_id,
                    c.number          AS chapter_number,
                    c.label           AS chapter_label,
                    m.id              AS material_id,
                    m.title           AS material_title,
                    m.cover_url       AS material_cover,
                    t.target_lang,
                    t.owner_id        AS creator_id,
                    u.display_name    AS creator_name,
                    {_ts('t.created_at')}
                FROM translations t
                JOIN chapters  c ON c.id = t.chapter_id
                JOIN materials m ON m.id = c.material_id
                LEFT JOIN users u ON u.id = t.owner_id
                LEFT JOIN translation_drafts d ON d.id = t.draft_id
                WHERE t.in_feed = TRUE
                  AND t.takedown_at IS NULL
                  AND (
                    t.feed_guild_id = $1
                    OR (
                      t.feed_guild_id IS NULL
                      AND d.visibility = 'all_guilds'
                      AND EXISTS (
                        SELECT 1 FROM user_guilds ug
                        WHERE ug.user_id = t.owner_id AND ug.guild_id = $1
                      )
                    )
                  )
                {cond_before}
                ORDER BY t.created_at DESC
                LIMIT $2
                """,
                *args,
            )
        return [dict(r) for r in rows]

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
        become re-claimable.
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

    async def requeue_task(
        self,
        target_kind: str, target_id: int,
        stage: str, error: str,
    ) -> None:
        """Mark a task as transient-failed; bump attempts but release
        the claim so the next worker can retry. Hitting MAX_TASK_ATTEMPTS
        leaves the row dead-lettered (claimed_by stays NULL, attempts
        won't decrease back to claimable)."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET claimed_by=NULL, claimed_at=NULL, "
                "  attempts=attempts+1, last_error=$1 "
                "WHERE target_kind=$2 AND target_id=$3 AND stage=$4",
                error, target_kind, target_id, stage,
            )

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

    # ── Chapter inbox (deferred prepare handle) ───────────────────

    async def set_inbox_handle(self, handle: "InboxHandle") -> None:
        parts_json = json.dumps(
            [{"number": p.number, "etag": p.etag} for p in handle.parts],
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO material_inbox "
                "  (chapter_id, tmp_id, upload_id, parts, title) "
                "VALUES ($1, $2, $3, $4::jsonb, $5) "
                "ON CONFLICT (chapter_id) DO UPDATE SET "
                "  tmp_id=EXCLUDED.tmp_id, upload_id=EXCLUDED.upload_id, "
                "  parts=EXCLUDED.parts, title=EXCLUDED.title",
                handle.chapter_id, handle.tmp_id, handle.upload_id,
                parts_json, handle.title,
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
        raw = row["parts"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        return InboxHandle(
            chapter_id=chapter_id,
            tmp_id=row["tmp_id"],
            upload_id=row["upload_id"],
            parts=tuple(
                CompletedPart(number=int(p["number"]), etag=str(p["etag"]))
                for p in raw
            ),
            title=row["title"],
        )

    async def clear_inbox_handle(self, chapter_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM material_inbox WHERE chapter_id=$1",
                chapter_id,
            )

    # ── DMCA ─────────────────────────────────────────────────────

    async def record_dmca_takedown(
        self,
        *,
        target_kind:    str,
        target_id:      int,
        scope_guild_id: str | None,
        reason:         str,
        reporter:       str,
    ) -> int:
        """Log + flip takedown_at on the target. Transactional so the
        log and the takedown either both succeed or both fail."""
        if target_kind not in ("material", "chapter", "draft", "translation"):
            raise ValueError(f"invalid target_kind: {target_kind!r}")
        table = {
            "material":    None,   # no takedown_at column; row deleted
            "chapter":     None,   # ditto
            "draft":       "translation_drafts",
            "translation": "translations",
        }[target_kind]
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "INSERT INTO dmca_takedowns "
                "  (target_kind, target_id, scope_guild_id, reason, reporter) "
                "VALUES ($1, $2, $3, $4, $5) RETURNING id",
                target_kind, target_id, scope_guild_id, reason, reporter,
            )
            if table is not None:
                await conn.execute(
                    f"UPDATE {table} SET takedown_at=NOW(), takedown_reason=$2 "
                    "WHERE id=$1",
                    target_id, reason,
                )
            else:
                # material / chapter: hard delete (no takedown_at marker)
                if target_kind == "material":
                    await conn.execute(
                        "DELETE FROM materials WHERE id=$1", target_id,
                    )
                else:
                    await conn.execute(
                        "DELETE FROM chapters WHERE id=$1", target_id,
                    )
        return row["id"]

    async def restore_dmca_takedown(self, takedown_id: int) -> bool:
        """Clear takedown markers. For materials/chapters that were
        hard-deleted, restoration is not possible — caller gets
        False and must re-import."""
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT target_kind, target_id FROM dmca_takedowns "
                "WHERE id=$1 AND restored_at IS NULL",
                takedown_id,
            )
            if row is None:
                return False
            tk = row["target_kind"]
            ti = row["target_id"]
            table = {
                "draft":       "translation_drafts",
                "translation": "translations",
            }.get(tk)
            if table is None:
                # material / chapter: already deleted; cannot restore.
                return False
            await conn.execute(
                f"UPDATE {table} SET takedown_at=NULL, takedown_reason=NULL "
                "WHERE id=$1",
                ti,
            )
            await conn.execute(
                "UPDATE dmca_takedowns SET restored_at=NOW() WHERE id=$1",
                takedown_id,
            )
        return True

    async def list_dmca_takedowns(
        self, *, active_only: bool = True, limit: int = 100,
    ) -> list[dict]:
        cond = "WHERE restored_at IS NULL" if active_only else ""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, target_kind, target_id, scope_guild_id, "
                "       reason, reporter, "
                f"       {_ts('taken_down_at')}, {_ts('restored_at')} "
                f"FROM dmca_takedowns {cond} "
                "ORDER BY taken_down_at DESC LIMIT $1",
                limit,
            )
        return [dict(r) for r in rows]
