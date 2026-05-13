"""Material routes.

A "material" is one source's manifestation of a manga — (source,
upstream_ref) for plugin-backed rows, per-row for ext / upload. The
user-facing manga page lives at `GET /work/{id}` (Work-centric); the
material routes here exist only for create / edit / delete of the
underlying source-bound row. Resolving the manga UI from a material
goes through `materials.work_id`.

Endpoints:
    POST   /api/material/import          source-backed lookup-or-create
    POST   /api/material                  ext / upload creation
    PATCH  /api/material/{id}             editable fields (ext/upload owner only)
    DELETE /api/material/{id}             cascade (ext/upload owner only)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import get_config, get_storage, get_store, require_user
from typoon.api.models import MaterialOut
from typoon.api.routes._shared import (
    require_material, require_material_admin,
)
from typoon.storage import Store

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api/material", tags=["material"],
    dependencies=[Depends(require_user)],
)


# ── Request bodies ────────────────────────────────────────────────────


class ImportBody(BaseModel):
    """Idempotent import for source-backed materials.

    The manifest runtime on the SPA already fetched the upstream page
    and resolved the display snapshot; we trust it for first-write but
    refresh on later imports via a background job (out of scope here).
    """
    source:        str
    upstream_ref:  str

    title:         str
    cover_url:     str | None = None
    description:   str | None = None
    author:        str | None = None
    status:        str | None = None
    languages:     list[str] = []
    title_native:  str | None = None
    title_alt:     list[str] = []
    cross_refs:    dict | None = None
    nsfw:          bool = False


class CreateLocalBody(BaseModel):
    origin:       str        # 'extension' | 'upload'
    title:        str
    cover_url:    str | None = None
    description:  str | None = None
    author:       str | None = None
    nsfw:         bool = False


class PatchMaterialBody(BaseModel):
    title:        str | None = None
    cover_url:    str | None = None
    description:  str | None = None
    nsfw:         bool | None = None


# ── Import / create ───────────────────────────────────────────────────


@router.post("/import", response_model=MaterialOut)
async def import_material(
    body: ImportBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Idempotent for (source, upstream_ref). A second call returns the
    same row id — `imported_by` is set on first insert only."""
    material_id = await db.get_or_create_source_material(
        source=body.source,
        upstream_ref=body.upstream_ref,
        title=body.title,
        cover_url=body.cover_url,
        description=body.description,
        author=body.author,
        status=body.status,
        languages=body.languages,
        title_native=body.title_native,
        title_alt=body.title_alt,
        cross_refs=body.cross_refs,
        nsfw=body.nsfw,
        imported_by=user["id"],
    )
    mat = await db.get_material(material_id)
    assert mat is not None
    return MaterialOut.from_row(mat)


@router.post("", response_model=MaterialOut)
async def create_local_material(
    body: CreateLocalBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Per-row material for extension capture or user upload. No dedup."""
    if body.origin not in ("extension", "upload"):
        raise HTTPException(
            400, "origin must be 'extension' or 'upload'",
        )
    material_id = await db.create_local_material(
        origin=body.origin,
        title=body.title,
        cover_url=body.cover_url,
        description=body.description,
        author=body.author,
        nsfw=body.nsfw,
        imported_by=user["id"],
    )
    mat = await db.get_material(material_id)
    assert mat is not None
    return MaterialOut.from_row(mat)


# ── Read ──────────────────────────────────────────────────────────────
#
# `GET /material/{id}` was removed in the schema 23 cutover; the canonical
# detail page lives at `GET /work/{id}` now (Step 2 of the Work refactor
# adds it). External callers should resolve material → work via the
# import response or the work payload.


# ── Edit ──────────────────────────────────────────────────────────────


@router.patch("/{material_id}", response_model=MaterialOut)
async def patch_material(
    material_id: int,
    body:        PatchMaterialBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Ext / upload owner can update display fields. Source-backed
    materials are immutable (their state mirrors the manifest)."""
    await require_material_admin(material_id, user, db)
    await db.update_material_metadata(
        material_id,
        title=body.title,
        cover_url=body.cover_url,
        description=body.description,
        nsfw=body.nsfw,
    )
    mat = await db.get_material(material_id)
    assert mat is not None
    return MaterialOut.from_row(mat)


# ── Cross-reference enrichment ───────────────────────────────────────
#
# Auto-enrich is client-driven: the SPA fans search across link
# plugins (Anilist, MAL, …) when it opens a Work whose cross_refs
# are empty, and POSTs whatever it found back here. This endpoint
# MERGES the suggested cross_refs additively — never overwrites a
# value that already exists. Anyone authenticated may submit; the
# operation is idempotent.
#
# After the merge we ask the linker to re-evaluate whether this
# material's Work should attach to an existing Work that shares a
# cross_ref namespace. The linker decides; this route doesn't merge
# anything itself.


class EnrichRefsBody(BaseModel):
    """Client-supplied cross_refs to merge into the material.

    `cross_refs` is `{namespace: value}` — same shape as the
    materials row column. Empty / non-scalar values are dropped
    server-side. The optional `source_signals` is logged for audit
    but doesn't affect the merge; it's there so a future moderation
    UI can see "which plugin claimed this".
    """
    cross_refs: dict[str, str | int | float]
    source_signals: list[dict] | None = None


@router.post(
    "/{material_id}/enrich-refs",
    response_model=MaterialOut,
)
async def enrich_material_refs(
    material_id: int,
    body:        EnrichRefsBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Merge client-found cross_refs onto a material additively.

    The SPA's link-plugin runtime calls this after fanning search out
    across services (Anilist, MAL, …). We accept whatever it found
    even when the same plugin disagrees with an earlier write —
    `merge_material_cross_refs` keeps the older value on conflict
    (anyone with stronger evidence will revote/re-merge through the
    community vote flow, not by overwriting here).
    """
    await require_user  # already enforced by Depends
    mat = await db.get_material(material_id)
    if mat is None:
        raise HTTPException(404, "Material not found")

    # Audit log — optional. Skipped at MVP; we keep the parameter so
    # the client signature is stable.
    _ = body.source_signals
    _ = user

    await db.merge_material_cross_refs(material_id, body.cross_refs)
    refreshed = await db.get_material(material_id)
    assert refreshed is not None
    return MaterialOut.from_row(refreshed)


# ── Delete ────────────────────────────────────────────────────────────


@router.delete("/{material_id}", status_code=204)
async def delete_material(
    material_id: int,
    user:   dict            = Depends(require_user),
    db:     Store           = Depends(get_store),
    stores: StorageRegistry = Depends(get_storage),
):
    """Ext / upload owner can delete; source-backed materials are
    immutable to users (DMCA admin path handles those).

    Cascade drops chapters → drafts → translations → bubbles etc. via
    FK ON DELETE CASCADE. We strip pipeline blobs (prepared, masks)
    and any render archives before the DB cascade, otherwise the
    locator pointers vanish first and we orphan blobs on remote
    storage.
    """
    from typoon.adapters.chapter_archive import masks_key, prepared_key

    await require_material_admin(material_id, user, db)
    cfg = get_config()
    salt = cfg.storage.archive_path_salt.encode()
    chapters = await db.list_chapters(material_id)
    for ch in chapters:
        await stores.pipeline.delete(prepared_key(ch["id"]))
        await stores.pipeline.delete(masks_key(ch["id"]))
        # Per-draft default renders + per-translation overrides each
        # live on whichever backend wrote them. Dispatch by backend so
        # multi-backend coexistence (e.g. mid-migration) doesn't leak
        # blobs.
        await _drop_archives_for_chapter(db, stores, ch["id"], salt=salt)

    await db.delete_material(material_id)


# ── Cleanup helpers ──────────────────────────────────────────────────


async def _drop_archives_for_chapter(
    db:      Store,
    stores:  StorageRegistry,
    chapter_id: int,
    *,
    salt:    bytes,
) -> None:
    """Remove every render archive for a chapter — draft defaults and
    per-translation overrides — across whichever backends wrote them.

    Failures are logged but not raised: an unreachable backend or
    already-gone blob must not block the DB cascade. The pipeline
    blobs (prepared/masks) are handled by the caller because their
    backend (always `stores.pipeline`) is fixed.
    """
    from typoon.adapters.chapter_archive import render_key

    drafts = await db.list_drafts_for_chapter(chapter_id)
    for d in drafts:
        await _drop_archive(
            stores,
            backend=d.get("archive_backend"),
            locator=d.get("archive_locator")
                    or render_key("draft", d["id"], salt),
            label=f"draft={d['id']}",
        )

    translations = await db.list_all_translations_for_chapter(chapter_id)
    for t in translations:
        await _drop_archive(
            stores,
            backend=t.get("archive_backend"),
            locator=t.get("archive_locator")
                    or render_key("translation", t["id"], salt),
            label=f"translation={t['id']}",
        )


async def _drop_archive(
    stores:  StorageRegistry,
    *,
    backend: str | None,
    locator: str | None,
    label:   str,
) -> None:
    """Best-effort delete of a single archive blob. Used by chapter /
    material cleanup; tolerates missing backend or stale locator."""
    if not backend or not locator:
        return
    try:
        reader = stores.reader(backend)
    except RuntimeError:
        logger.warning(
            "skip %s archive cleanup: backend %r not configured",
            label, backend,
        )
        return
    try:
        await reader.delete(locator)
    except Exception:
        logger.exception(
            "archive cleanup failed for %s (backend=%s locator=%s)",
            label, backend, locator,
        )
