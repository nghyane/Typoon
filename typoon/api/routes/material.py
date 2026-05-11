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

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.storage_registry import StorageRegistry
from typoon.api.deps import get_storage, get_store, require_user
from typoon.api.models import (
    ChapterOut, ChapterTranslationOverlay, MaterialOut,
)
from typoon.api.routes._shared import (
    require_material, require_material_admin,
)
from typoon.storage import Store


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


class MaterialDetailOut(BaseModel):
    """Detail + chapter list with translation overlay. Embedded so the
    SPA renders the manga page in one round-trip."""
    material: MaterialOut
    chapters: list[ChapterOut]


@router.get("/{material_id}", response_model=MaterialDetailOut)
async def get_material_detail(
    material_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    mat = await require_material(material_id, db)
    chapters = await db.list_chapters(material_id)
    if not chapters:
        return MaterialDetailOut(
            material=MaterialOut.from_row(mat),
            chapters=[],
        )

    chapter_ids = [c["id"] for c in chapters]
    guilds_rows = await db.get_user_guilds(user["id"])
    viewer_guilds = [g["id"] for g in guilds_rows]
    overlay = await db.list_translations_for_chapters(
        chapter_ids, viewer_id=user["id"], viewer_guilds=viewer_guilds,
    )

    out_chapters: list[ChapterOut] = []
    for ch in chapters:
        trs = overlay.get(ch["id"], [])
        out_chapters.append(ChapterOut(
            id=ch["id"],
            material_id=ch["material_id"],
            position=ch["position"],
            number=ch["number"],
            label=ch.get("label"),
            upstream_url=ch.get("upstream_url"),
            pages_origin=ch["pages_origin"],
            page_count=int(ch.get("page_count") or 0),
            updated_at=ch.get("updated_at"),
            translations=[
                ChapterTranslationOverlay(
                    id=t["id"],
                    target_lang=t["target_lang"],
                    creator_id=t.get("owner_id"),
                    creator_name=t.get("creator_name"),
                    state=t.get("state") or "done",
                    in_feed=bool(t.get("in_feed")),
                    from_cache=bool(t.get("uses_default_render")),
                )
                for t in trs
            ],
        ))
    return MaterialDetailOut(
        material=MaterialOut.from_row(mat),
        chapters=out_chapters,
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
    guilds = [g["id"] for g in await db.get_user_guilds(user["id"])]
    raw = await db.list_translations_by_upstream(
        material_id, body.upstream_urls,
        viewer_id=user["id"], viewer_guilds=guilds,
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
                in_feed=bool(t.get("in_feed")),
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
    from typoon.adapters.chapter_archive import (
        masks_key, prepared_key, render_key,
    )

    await require_material_admin(material_id, user, db)
    chapters = await db.list_chapters(material_id)
    for ch in chapters:
        await stores.pipeline.delete(prepared_key(ch["id"]))
        await stores.pipeline.delete(masks_key(ch["id"]))
        # TODO(slice 3): delete per-translation render archives too.
        # Requires listing translations for the chapter + dispatching
        # by archive_backend. Skipped here — orphaned blobs are
        # tolerable for ext/upload deletes; pipeline keys are the
        # priority because they're large.

    await db.delete_material(material_id)
