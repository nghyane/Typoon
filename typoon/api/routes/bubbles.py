"""Bubble OCR + translation routes.

Read view (GET) merges bubbles and translations into a single list per
chapter. Edit view (PATCH) updates one translation row and re-enqueues
render so the page picture catches up.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.deps import get_store, require_user
from typoon.api.models import BubbleOut
from typoon.api.routes._shared import (
    require_chapter, require_project_owner, require_project_view,
)
from typoon.storage import Store

router = APIRouter(
    prefix="/api/projects", tags=["bubbles"],
    dependencies=[Depends(require_user)],
)


class TranslationPatch(BaseModel):
    translated_text: str
    kind:            str | None = None  # dialogue | sfx | skip


@router.get(
    "/{project_id}/chapters/{chapter_id}/bubbles",
    response_model=list[BubbleOut],
)
async def list_bubbles(
    project_id: int,
    chapter_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """All bubbles in chapter — OCR text plus translation if available."""
    await require_project_view(project_id, user, db)
    await require_chapter(project_id, chapter_id, db)
    bubbles = await db.get_bubbles(chapter_id)
    translations = await db.get_translations(chapter_id)
    out: list[BubbleOut] = []
    for b in bubbles:
        key = (b["page_index"], b["bubble_idx"])
        t = translations.get(key)
        out.append(BubbleOut(
            page_index=b["page_index"],
            bubble_idx=b["bubble_idx"],
            source_text=b["source_text"],
            translated_text=t["translated_text"] if t else None,
            kind=t["kind"] if t else None,
            confidence=b["confidence"],
        ))
    return out


@router.patch(
    "/{project_id}/chapters/{chapter_id}/bubbles/{page_index}/{bubble_idx}",
    response_model=BubbleOut,
)
async def patch_translation(
    project_id: int,
    chapter_id: int,
    page_index: int,
    bubble_idx: int,
    body:       TranslationPatch,
    user:       dict  = Depends(require_user),
    db:         Store = Depends(get_store),
):
    """Manual edit. Owner only — re-enqueues render."""
    await require_project_owner(project_id, user, db)
    await require_chapter(project_id, chapter_id, db)
    if body.kind is not None and body.kind not in ("dialogue", "sfx", "skip"):
        raise HTTPException(400, "kind must be one of: dialogue, sfx, skip")
    updated = await db.update_translation(
        chapter_id, page_index, bubble_idx,
        body.translated_text, body.kind,
    )
    if not updated:
        raise HTTPException(404, "Bubble not found or not yet translated")

    # Render gets a fresh task; downstream worker will pick it up.
    await db.enqueue(chapter_id, "render")

    bubbles = {
        (b["page_index"], b["bubble_idx"]): b
        for b in await db.get_bubbles(chapter_id)
    }
    b = bubbles.get((page_index, bubble_idx))
    if b is None:
        raise HTTPException(404, "Bubble row missing after update")
    translations = await db.get_translations(chapter_id)
    t = translations.get((page_index, bubble_idx))
    return BubbleOut(
        page_index=page_index,
        bubble_idx=bubble_idx,
        source_text=b["source_text"],
        translated_text=t["translated_text"] if t else None,
        kind=t["kind"] if t else None,
        confidence=b["confidence"],
    )
