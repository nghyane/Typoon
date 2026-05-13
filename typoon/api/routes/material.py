"""Material routes.

Materials are the unit a user "follows" — manga identity, source-
agnostic. Source-backed materials are cross-user (idempotent import);
ext / upload are per-row.

Endpoints:
    POST   /api/material/import          source-backed lookup-or-create
    POST   /api/material                  ext / upload creation
    GET    /api/material/{id}             detail + chapters + translations overlay
    PATCH  /api/material/{id}             editable fields (ext/upload owner only)
    DELETE /api/material/{id}             cascade (ext/upload owner only)

The chapter list overlay embeds per-chapter translation summary the
viewer is authorized to see (see Store.list_translations_for_chapters).
The reader does not need a second round-trip to know which translations
exist for a chapter.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import get_config, get_storage, get_store, require_user
from typoon.api.models import (
    ChapterOut, ChapterTranslationOverlay, MaterialOut,
)
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


class ViewerLibraryRef(BaseModel):
    """The viewer's library entry that links this material, if any.

    Lets `/material/$id` decide between "Theo dõi" (no entry yet) and
    "Mở trong Library" (jump to the per-user `/title/$entryId` page)
    without a second round-trip.
    """
    entry_id: int
    status:   str  # LibraryStatus literal; kept loose for forward-compat


class MaterialDetailOut(BaseModel):
    """Detail + chapter list with translation overlay. Embedded so the
    SPA renders the manga page in one round-trip."""
    material:     MaterialOut
    chapters:     list[ChapterOut]
    viewer_entry: ViewerLibraryRef | None = None


@router.get("/{material_id}", response_model=MaterialDetailOut)
async def get_material_detail(
    material_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    mat = await require_material(material_id, db)
    viewer_entry_row = await db.find_entry_for_material(
        user_id=user["id"], material_id=material_id,
    )
    viewer_entry = (
        ViewerLibraryRef(
            entry_id=viewer_entry_row["id"],
            status=viewer_entry_row["status"],
        )
        if viewer_entry_row else None
    )

    chapters = await db.list_chapters(material_id)
    if not chapters:
        return MaterialDetailOut(
            material=MaterialOut.from_row(mat),
            chapters=[],
            viewer_entry=viewer_entry,
        )

    chapter_ids = [c["id"] for c in chapters]
    overlay = await db.list_translations_for_chapters(chapter_ids)

    out_chapters: list[ChapterOut] = []
    for ch in chapters:
        trs = overlay.get(ch["id"], [])
        out_chapters.append(ChapterOut(
            id=ch["id"],
            material_id=ch["material_id"],
            position=ch["position"],
            number=ch["number_norm"],
            label=ch.get("label"),
            upstream_url=ch.get("upstream_url"),
            page_count=int(ch.get("page_count") or 0),
            updated_at=ch.get("updated_at"),
            translations=[
                ChapterTranslationOverlay(
                    id=t["id"],
                    target_lang=t["target_lang"],
                    creator_id=t.get("owner_id"),
                    creator_name=t.get("creator_name"),
                    state=t.get("state") or "done",
                    from_cache=bool(t.get("uses_default_render")),
                )
                for t in trs
            ],
        ))
    return MaterialDetailOut(
        material=MaterialOut.from_row(mat),
        chapters=out_chapters,
        viewer_entry=viewer_entry,
    )


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


# ── Manifest-side overlay ─────────────────────────────────────────────


class TranslationOverlayBody(BaseModel):
    upstream_urls: list[str]


@router.post(
    "/{material_id}/translation-overlay",
    response_model=dict[str, list[ChapterTranslationOverlay]],
)
async def translation_overlay(
    material_id: int,
    body:        TranslationOverlayBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Bulk overlay query for the MangaPage chapter list.

    The SPA has the manifest chapter list (upstream URLs); we return
    which of those chapters already have visible translations so the
    UI can render "[VN] @nghyane Đọc" instead of "Dịch" for cached
    rows. Used POST so we can ship a long list without URL bloat;
    semantics are still pure-read.
    """
    if not body.upstream_urls:
        return {}
    await require_material(material_id, db)
    raw = await db.list_translations_by_upstream(
        material_id, body.upstream_urls,
    )
    out: dict[str, list[ChapterTranslationOverlay]] = {}
    for url, trs in raw.items():
        out[url] = [
            ChapterTranslationOverlay(
                id=t["id"],
                target_lang=t["target_lang"],
                creator_id=t.get("owner_id"),
                creator_name=t.get("creator_name"),
                state=t.get("state") or "done",
                from_cache=bool(t.get("uses_default_render")),
            )
            for t in trs
        ]
    return out


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
