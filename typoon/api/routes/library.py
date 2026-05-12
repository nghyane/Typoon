"""Library routes — per-user grouping of materials.

Endpoints:
    GET    /api/library                     entries with linked materials
    POST   /api/library/entry               create entry from a material
    GET    /api/library/entry/{id}          one entry
    PATCH  /api/library/entry/{id}          title, status, target_lang,
                                            auto_translate, last-read
    DELETE /api/library/entry/{id}          delete + unlink materials

    POST   /api/library/entry/{id}/link     attach material (casts +1)
    POST   /api/library/entry/{id}/unlink   detach material (removes vote)

    GET    /api/library/suggest?material_id=  cross-source suggestion
    POST   /api/library/suggest/{candidate}/reject?material_id=
                                              -1 vote on pair

Reading status, last-read, "Tiếp tục đọc" all live on the entry —
never on the material directly.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from typoon.api.deps import get_store, require_user
from typoon.api.models import (
    LibraryEntryOut, LibraryMaterialLink, LibrarySuggestionOut,
    LibraryStatus, TranslationSummary,
)
from typoon.api.routes._shared import require_library_entry, require_material
from typoon.storage import Store


router = APIRouter(
    prefix="/api/library", tags=["library"],
    dependencies=[Depends(require_user)],
)


def _entry_to_out(row: dict) -> LibraryEntryOut:
    summary = row.get("translation_summary") or {}
    return LibraryEntryOut(
        id=row["id"],
        title=row["title"],
        cover_url=row.get("cover_url"),
        primary_material_id=row.get("primary_material_id"),
        status=row["status"],
        target_lang=row.get("target_lang"),
        auto_translate=bool(row.get("auto_translate")),
        last_read_at=row.get("last_read_at"),
        last_chapter_ref=row.get("last_chapter_ref"),
        materials=[
            LibraryMaterialLink(
                material_id=m["material_id"],
                link_origin=m["link_origin"],
                linked_at=m.get("linked_at"),
            )
            for m in (row.get("materials") or [])
        ],
        translation_summary=TranslationSummary(
            pending=int(summary.get("pending", 0)),
            running=int(summary.get("running", 0)),
            done=int(summary.get("done", 0)),
            error=int(summary.get("error", 0)),
        ),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


# ── List / read ───────────────────────────────────────────────────────


@router.get("", response_model=list[LibraryEntryOut])
async def list_entries(
    status: LibraryStatus | None = Query(None),
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """List entries filtered by status. Default (no status) hides
    `dropped` so users don't see things they explicitly removed."""
    rows = await db.list_library_entries(user["id"], status=status)
    return [_entry_to_out(r) for r in rows]


@router.get("/entry/{entry_id}", response_model=LibraryEntryOut)
async def get_entry(
    entry_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    row = await require_library_entry(entry_id, user, db)
    return _entry_to_out(row)


# ── Create / update / delete ──────────────────────────────────────────


class CreateEntryBody(BaseModel):
    material_id:    int
    title:          str | None = None
    cover_url:      str | None = None
    # Optional reading preference baked in at Add time. UI prompts for
    # these in the Add-manga modal; defaults preserve the casual case
    # (status=reading, target_lang=None means "ask later").
    target_lang:    str | None       = None
    auto_translate: bool             = False
    status:         LibraryStatus    = "reading"


@router.post("/entry", response_model=LibraryEntryOut)
async def create_entry(
    body: CreateEntryBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    mat = await require_material(body.material_id, db)
    entry_id = await db.create_library_entry(
        user_id=user["id"],
        title=body.title or mat["title"],
        cover_url=body.cover_url or mat.get("cover_url"),
        primary_material_id=body.material_id,
        target_lang=body.target_lang,
        auto_translate=body.auto_translate,
        status=body.status,
    )
    row = await db.get_library_entry(entry_id, user["id"])
    assert row is not None
    return _entry_to_out(row)


class PatchEntryBody(BaseModel):
    title:            str | None           = None
    status:           LibraryStatus | None = None
    target_lang:      str | None           = None
    auto_translate:   bool | None          = None
    last_read_at:     str | None           = None
    last_chapter_ref: dict | None          = None


@router.patch("/entry/{entry_id}", response_model=LibraryEntryOut)
async def patch_entry(
    entry_id: int,
    body:     PatchEntryBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    await require_library_entry(entry_id, user, db)
    await db.update_library_entry(
        entry_id, user["id"],
        title=body.title,
        status=body.status,
        target_lang=body.target_lang,
        auto_translate=body.auto_translate,
        last_read_at=body.last_read_at,
        last_chapter_ref=body.last_chapter_ref,
    )
    row = await db.get_library_entry(entry_id, user["id"])
    assert row is not None
    return _entry_to_out(row)


@router.delete("/entry/{entry_id}", status_code=204)
async def delete_entry(
    entry_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    ok = await db.delete_library_entry(entry_id, user["id"])
    if not ok:
        raise HTTPException(404, "Library entry not found")


# ── Material linking ──────────────────────────────────────────────────


class LinkBody(BaseModel):
    material_id:  int
    # 'auto' = system suggested + user confirmed
    # 'manual' = user typed it in via search
    link_origin:  str = "manual"


@router.post("/entry/{entry_id}/link", status_code=204)
async def link_material(
    entry_id: int,
    body:     LinkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Attach a material to an existing entry. Casts +1 vote on every
    pair (existing material in entry, new material). The vote is what
    powers community-driven cross-source suggestions for other users."""
    await require_library_entry(entry_id, user, db)
    await require_material(body.material_id, db)
    if body.link_origin not in ("primary", "auto", "manual"):
        raise HTTPException(400, "invalid link_origin")
    await db.link_material_to_entry(
        entry_id=entry_id,
        material_id=body.material_id,
        link_origin=body.link_origin,
        voter_id=user["id"],
    )


class UnlinkBody(BaseModel):
    material_id: int


@router.post("/entry/{entry_id}/unlink", status_code=204)
async def unlink_material(
    entry_id: int,
    body:     UnlinkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Detach a material from an entry. Removes the voter's +1 votes
    on every pair involving this material in the entry. If the entry
    has no materials left, callers should DELETE the entry separately
    (we do not auto-delete to avoid surprising the user with vanished
    bookmark state)."""
    await require_library_entry(entry_id, user, db)
    await db.unlink_material_from_entry(
        entry_id=entry_id,
        material_id=body.material_id,
        voter_id=user["id"],
    )


# ── Suggestion (cross-source linking) ─────────────────────────────────


@router.get("/suggest", response_model=LibrarySuggestionOut | None)
async def suggest(
    material_id: int = Query(...),
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Returns 0-or-1 candidate entry the SPA should offer to merge
    into. Ranking per RFC §7.4.1: cross_refs > vote≥3 > title_native >
    vote 1-2 > none. Returns None when no signal fires."""
    await require_material(material_id, db)
    out = await db.find_library_suggestion(
        user_id=user["id"], material_id=material_id,
    )
    if out is None:
        return None
    return LibrarySuggestionOut(**out)


class RejectBody(BaseModel):
    material_id: int                # the one user is currently viewing
    candidate_material_id: int      # the one we suggested they merge into


@router.post("/suggest/reject", status_code=204)
async def reject_suggestion(
    body: RejectBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """User clicked "Không, manga khác" — casts -1 vote so the same
    pair won't surface again for this user, and weakens the signal
    for everyone else."""
    await require_material(body.material_id, db)
    await require_material(body.candidate_material_id, db)
    await db.reject_library_suggestion(
        voter_id=user["id"],
        material_id=body.material_id,
        candidate_material_id=body.candidate_material_id,
    )
