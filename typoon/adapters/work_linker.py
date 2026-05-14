"""Work auto-link adapter.

When a material is created, the linker picks the canonical Work it
belongs to based on `cross_refs`. The intent ("Cách 1 — danh bạ"):
`works` is identity only; per-source display data stays on the
material. The linker decides solely whether two materials share a
Work, never copies metadata.

Rules
-----

Match  — at least one (namespace, value) pair in `cross_refs` overlaps
         with an existing Work's `cross_refs`. Attach to that Work.

Conflict — any namespace has a value present on a candidate Work but
           differs from the new material's value. Treat as identity
           collision: do NOT attach; create a fresh isolated Work
           (community vote may merge them later).

No match / no cross_refs — create an isolated Work (1-1 with this
                            material) and attach.

The linker is pure SQL inside one transaction so concurrent imports
of the same upstream cannot race past the cross_refs lookup. Callers
hold the asyncpg connection.
"""

from __future__ import annotations

import asyncpg


async def link_or_create_work(
    conn: asyncpg.Connection,
    cross_refs: dict | None,
) -> int:
    """Resolve the Work id for a material being created.

    Caller passes the live connection so this can run inside the same
    transaction that inserts the material — keeps the (link, insert)
    pair atomic. Returns the chosen `works.id`.

    `cross_refs` is the dict as it appears on the material payload —
    `{"mdex_uuid": "...", "anilist": 12345}` or similar. Empty / None
    falls through to "create isolated Work".
    """
    refs = _clean_refs(cross_refs)
    if not refs:
        return await _create_work(conn, None)

    matched_id = await find_compatible_work(conn, refs)
    if matched_id is not None:
        # Augment the Work's cross_refs with any namespaces the new
        # material brought along — purely additive, never overwrites.
        await _merge_refs(conn, matched_id, refs)
        return matched_id

    return await _create_work(conn, refs)


async def find_compatible_work(
    conn:     asyncpg.Connection,
    refs:     dict[str, str],
    *,
    exclude:  int | None = None,
) -> int | None:
    """Pick the oldest existing Work whose `cross_refs` are compatible
    with `refs` (share at least one namespace, no conflicting values).
    Returns None when nothing matches.

    Set `exclude` to the caller's own work id when re-linking an
    EXISTING material whose cross_refs just expanded — we don't want
    to "match" the material's current Work and then attempt a no-op
    merge with itself.
    """
    if not refs:
        return None
    overlap_rows = await conn.fetch(
        "SELECT id, cross_refs FROM works "
        "WHERE cross_refs IS NOT NULL "
        "  AND cross_refs ?| $1::text[] "
        "  AND ($2::bigint IS NULL OR id <> $2) "
        "ORDER BY id ASC",
        list(refs.keys()),
        exclude,
    )
    for row in overlap_rows:
        existing = row["cross_refs"] or {}
        if _is_conflict(existing, refs):
            continue
        return int(row["id"])
    return None


# ── Helpers ───────────────────────────────────────────────────────────


def _clean_refs(cross_refs: dict | None) -> dict[str, str]:
    """Coerce cross_refs to {str: str}, dropping empty / non-scalar
    entries. Numbers come back as strings so JSONB equality stays
    deterministic (Postgres compares JSONB values structurally; mixing
    `12345` and `"12345"` would fail an else-equal namespace check).
    """
    if not cross_refs:
        return {}
    out: dict[str, str] = {}
    for k, v in cross_refs.items():
        if not k or v is None:
            continue
        if isinstance(v, (str, int, float)):
            s = str(v).strip()
            if s:
                out[str(k)] = s
    return out


def _is_conflict(existing: dict, new: dict[str, str]) -> bool:
    """Two refs conflict when ANY shared namespace has different
    string-coerced values. Disjoint namespaces never conflict — they
    augment each other.
    """
    for k, v in new.items():
        if k in existing:
            ev = existing[k]
            if ev is None:
                continue
            if str(ev).strip() != v:
                return True
    return False


async def _create_work(
    conn: asyncpg.Connection, cross_refs: dict[str, str] | None,
) -> int:
    row = await conn.fetchrow(
        "INSERT INTO works (cross_refs) VALUES ($1::jsonb) RETURNING id",
        cross_refs if cross_refs else None,
    )
    return int(row["id"])


async def _merge_refs(
    conn: asyncpg.Connection,
    work_id: int,
    new_refs: dict[str, str],
) -> None:
    """Add namespaces from `new_refs` to the Work's cross_refs without
    overwriting existing values. Postgres `||` does last-write-wins
    on JSONB, so we coalesce with the existing map on the LEFT to
    preserve the Work's authoritative values where present.
    """
    if not new_refs:
        return
    await conn.execute(
        "UPDATE works SET "
        "  cross_refs = COALESCE($2::jsonb, '{}'::jsonb) "
        "             || COALESCE(cross_refs, '{}'::jsonb), "
        "  updated_at = NOW() "
        "WHERE id = $1",
        work_id, new_refs,
    )
