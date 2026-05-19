"""Library routes — per-user, per-Work bookmark.

One entry per (user, Work). Adding a second material that auto-links
to the same Work slots into the existing entry via library_materials,
not a duplicate entry.

Endpoints:
    GET    /api/library                       entries with linked materials
    POST   /api/library/entry                  create entry for a Work
    GET    /api/library/entry/{id}             one entry
    PATCH  /api/library/entry/{id}             title, status
    DELETE /api/library/entry/{id}             delete + unlink materials

    POST   /api/library/entry/{id}/link        attach material (casts +1)
    POST   /api/library/entry/{id}/unlink      detach material (removes vote)

Reading history ("what chapter did I last open") lives in the
separate `reading_history` table — see /api/me/recent-reads.

Cross-source SUGGESTIONS (Commit 3): will live behind a Work-aware
signal cascade. The pre-Work suggestion path was removed with the
schema 23 cutover — there is no `find_library_suggestion` anymore.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.deps import get_store, require_user
from typoon.api.models import (
    LibraryEntryOut, LibraryMaterialLink,
    LibraryStatus, TranslationSummary,
)
from typoon.api.routes._shared import require_library_entry, require_material
from typoon.storage.store import Store


router = APIRouter(
    prefix="/api/library", tags=["library"],
    dependencies=[Depends(require_user)],
)


def _entry_to_out(row: dict) -> LibraryEntryOut:
    summary = row.get("translation_summary") or {}
    return LibraryEntryOut(
        id=row["id"],
        # `_resolve_work_display` stamps `title` + `cover` server-side,
        # biased by the viewer's reading lang. Same canonical name the
        # Work hub renders — never the raw snapshot a manifest first
        # imported.
        title=row.get("title") or "",
        cover_url=row.get("cover"),
        work_id=int(row["work_id"]),
        target_lang=row["target_lang"],
        status=row["status"],
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


@router.get("", response_model=list[LibraryEntryOut])
async def list_entries(
    status: LibraryStatus | None = None,
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


class CreateEntryBody(BaseModel):
    """Create a library entry for a Work.

    The Work is identified by `material_id` — the SPA passes whichever
    material the user is currently looking at; the server resolves
    its `work_id` and dedupes against any existing (user, Work) entry.
    The material itself is attached as `link_origin='manual'` so it
    shows up under the entry alongside any sibling materials linked
    later.

    No `title` / `cover_url` here — they live on the material(s)
    attached to the Work and resolve server-side at read time.
    """
    material_id:    int
    target_lang:    str           = "vi"
    status:         LibraryStatus = "reading"


@router.post("/entry", response_model=LibraryEntryOut)
async def create_entry(
    body: CreateEntryBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    mat = await require_material(body.material_id, db)
    work_id = int(mat["work_id"])
    # Idempotent per (user, Work). If an entry already exists, ensure
    # the requesting material is attached (handles "Theo dõi from a
    # new source after the Work was already followed via another").
    existing = await db.find_entry_for_work(
        user_id=user["id"], work_id=work_id,
    )
    if existing is not None:
        entry_id = int(existing["id"])
        await db.link_material_to_entry(
            entry_id=entry_id,
            material_id=body.material_id,
            link_origin="manual",
            voter_id=user["id"],
        )
    else:
        entry_id = await db.create_library_entry(
            user_id=user["id"],
            work_id=work_id,
            target_lang=body.target_lang,
            materials=[(body.material_id, "manual")],
            status=body.status,
        )
    row = await db.get_library_entry(entry_id, user["id"])
    assert row is not None
    return _entry_to_out(row)


class PatchEntryBody(BaseModel):
    status:      LibraryStatus | None = None
    target_lang: str | None           = None


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
        status=body.status,
        target_lang=body.target_lang,
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


class LinkBody(BaseModel):
    material_id:  int
    # 'auto'   = system suggested + user confirmed (e.g. cross_refs match)
    # 'manual' = user typed it in via search / explicit attach
    link_origin:  str = "manual"


@router.post("/entry/{entry_id}/link", status_code=204)
async def link_material(
    entry_id: int,
    body:     LinkBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Attach a material to an existing entry. Casts +1 vote on every
    pair (existing material in entry, new material). The vote powers
    community-driven cross-source suggestions for other users."""
    await require_library_entry(entry_id, user, db)
    await require_material(body.material_id, db)
    if body.link_origin not in ("auto", "manual"):
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
