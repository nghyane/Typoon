"""DMCA admin routes — takedown + restore.

User-facing report endpoint logs incoming complaints; admin endpoints
flip `takedown_at` flags or restore.

Per RFC §7.6 the takedown affects different targets differently:
  - 'draft' / 'translation' → set takedown_at, content becomes
                                invisible to readers + excluded from
                                cache lookup.
  - 'material' / 'chapter'  → hard delete (cascades blobs).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from typoon.api.deps import get_store, require_admin, require_user
from typoon.api.models import DmcaTakedownOut
from typoon.storage import Store


router = APIRouter(prefix="/api/dmca", tags=["dmca"])


# ── User-facing report ────────────────────────────────────────────────


class ReportBody(BaseModel):
    target_kind:    str               # 'material' | 'chapter' | 'draft' | 'translation'
    target_id:      int
    scope_guild_id: str | None = None
    reason:         str
    reporter:       str               # email / handle / discord username
    # `notice_text` (full DMCA notice body) is captured client-side via
    # form upload — out of scope for this minimal endpoint.


@router.post("/report", status_code=202)
async def report(
    body: ReportBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Anyone authenticated can report — this is the public-facing
    intake. Admin reviews via `/api/admin/dmca`. We do not auto-
    takedown on report; admin must confirm via the action endpoint.

    A future hardening pass may auto-mute (set in_feed=FALSE) on
    report pending review, but that's product-call, not architecture.
    """
    # For now we record the takedown directly as evidence trail; admin
    # decides on restore vs keep.
    if body.target_kind not in ("material", "chapter", "draft", "translation"):
        raise HTTPException(400, "invalid target_kind")
    takedown_id = await db.record_dmca_takedown(
        target_kind=body.target_kind,
        target_id=body.target_id,
        scope_guild_id=body.scope_guild_id,
        reason=body.reason,
        reporter=body.reporter,
    )
    return {"takedown_id": takedown_id}


# ── Admin ─────────────────────────────────────────────────────────────


admin_router = APIRouter(
    prefix="/api/admin/dmca", tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@admin_router.get("", response_model=list[DmcaTakedownOut])
async def list_admin(
    active_only: bool = Query(True),
    limit:       int  = Query(100, ge=1, le=500),
    db: Store = Depends(get_store),
):
    rows = await db.list_dmca_takedowns(
        active_only=active_only, limit=limit,
    )
    return [DmcaTakedownOut(**r) for r in rows]


@admin_router.post("/{takedown_id}/restore", status_code=204)
async def restore(
    takedown_id: int,
    db: Store = Depends(get_store),
):
    """Reverse a takedown. Only works for draft / translation targets;
    material / chapter takedowns are hard deletes and not restorable."""
    ok = await db.restore_dmca_takedown(takedown_id)
    if not ok:
        raise HTTPException(
            409,
            "Cannot restore — material/chapter takedowns are permanent",
        )
