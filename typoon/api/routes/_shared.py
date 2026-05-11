"""Shared route helpers — material / chapter / translation access checks.

Patterns:
  - `require_material` resolves and 404s.
  - `require_chapter` ensures chapter belongs to material.
  - `require_translation` resolves a per-user wrapper; visibility gate
    composes with the draft's visibility for cross-user reads.
  - `require_material_admin` checks ownership for ext / upload origins
    (source-backed materials have no per-user admin).
"""

from __future__ import annotations

from fastapi import HTTPException

from typoon.storage import Store


# ── Material ──────────────────────────────────────────────────────────


async def require_material(material_id: int, db: Store) -> dict:
    """Resolve material by id or 404. Source-backed materials are
    cross-user readable; ext / upload are also readable by any
    authenticated user (privacy isolation is at translation level)."""
    mat = await db.get_material(material_id)
    if mat is None:
        raise HTTPException(404, "Material not found")
    return mat


async def require_material_admin(
    material_id: int, user: dict, db: Store,
) -> dict:
    """Mutation gate. For ext / upload, only `imported_by` may edit /
    delete. Source-backed materials are not user-editable (their state
    is owned by the manifest snapshot)."""
    mat = await require_material(material_id, db)
    if mat["origin"] == "source":
        raise HTTPException(
            403, "Source-backed material is not editable",
        )
    if mat.get("imported_by") != user["id"]:
        raise HTTPException(403, "Only the importer can do this")
    return mat


# ── Chapter ──────────────────────────────────────────────────────────


async def require_chapter(
    material_id: int, chapter_id: int, db: Store,
) -> dict:
    ch = await db.get_chapter(chapter_id)
    if ch is None or ch["material_id"] != material_id:
        raise HTTPException(404, "Chapter not found")
    return ch


# ── Translation ──────────────────────────────────────────────────────


async def require_translation_owner(
    translation_id: int, user: dict, db: Store,
) -> dict:
    """Mutation gate for translation endpoints (edit, share toggle,
    redo, delete). Owner only."""
    t = await db.get_translation(translation_id)
    if t is None:
        raise HTTPException(404, "Translation not found")
    if t["owner_id"] != user["id"]:
        raise HTTPException(403, "Only the owner can do this")
    return t


# ── Library entry ────────────────────────────────────────────────────


async def require_library_entry(
    entry_id: int, user: dict, db: Store,
) -> dict:
    e = await db.get_library_entry(entry_id, user["id"])
    if e is None:
        raise HTTPException(404, "Library entry not found")
    return e
