"""Shared route helpers — material / chapter / translation access checks.

Patterns:
  - `require_material` resolves and 404s.
  - `require_chapter` ensures chapter belongs to material.
  - `require_translation` resolves a per-user wrapper; visibility gate
    composes with the draft's visibility for cross-user reads.
  - `require_material_admin` checks ownership for ext / upload origins
    (source-backed materials have no per-user admin).
  - `resolve_archive_url` builds the public CDN URL for a translation,
    handling the translation→draft fallback chain in one place so the
    rule lives next to its only consumers (translate, feed).
"""

from __future__ import annotations

from fastapi import HTTPException

from typoon.adapters.storage_registry import StorageRegistry
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


async def require_work(work_id: int, db: Store) -> dict:
    """Resolve a Work by id or 404. Works are cross-user identity hubs
    (no ownership), so the gate is "exists" only.

    If the requested id has been merged into another Work via
    community-vote (`work_redirects`), raise a structured 410 so the
    client can replace the URL in-place. Every route that touches a
    Work flows through here, so redirect handling is uniform —
    no per-endpoint plumbing.
    """
    work = await db.get_work(work_id)
    if work is not None:
        return work
    new_id = await db.get_work_redirect(work_id)
    if new_id is not None:
        raise HTTPException(
            status_code=410,
            detail={
                "kind":           "work_redirected",
                "requested_id":   work_id,
                "redirected_to":  new_id,
            },
        )
    raise HTTPException(404, "Work not found")


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


# ── Archive URL ──────────────────────────────────────────────────────


def _url_for(
    *,
    backend:     str | None,
    locator:     str | None,
    rendered_at: str | None,
    stores:      StorageRegistry,
) -> str | None:
    """Dispatch a (backend, locator) pair through the matching reader.

    Returns the public CDN URL or None when:
      - the row has no archive yet (backend/locator missing), or
      - the backend that wrote it is no longer configured (a coexistence
        artefact during a backend migration — silently treated as "no
        archive available" rather than 500).
    """
    if not backend or not locator:
        return None
    try:
        reader = stores.reader(backend)
    except RuntimeError:
        return None
    return reader.url(locator, version=rendered_at or "")


async def resolve_archive_url(
    t: dict, *,
    db:     Store,
    stores: StorageRegistry,
) -> str | None:
    """Build the public archive URL for a translation row, falling
    back to its draft's default render when the translation has no
    per-row archive of its own. Used by both `/api/translate/{id}`
    and `/api/feed/guild/{id}` so the fallback rule stays single-
    sourced."""
    if t.get("archive_locator"):
        return _url_for(
            backend=t.get("archive_backend"),
            locator=t.get("archive_locator"),
            rendered_at=t.get("rendered_at"),
            stores=stores,
        )
    draft_id = t.get("draft_id")
    if not draft_id:
        return None
    draft = await db.get_draft(draft_id)
    if draft is None:
        return None
    return _url_for(
        backend=draft.get("archive_backend"),
        locator=draft.get("archive_locator"),
        rendered_at=draft.get("rendered_at") or t.get("rendered_at"),
        stores=stores,
    )
