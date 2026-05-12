"""Translator memory routes.

Memory is the v2 replacement for project-cũ settings: per (user,
material, target_lang) cards (characters / world / style / glossary /
style_refs) plus an accumulated sliding window of chapter briefs.

Endpoints:
    GET    /api/material/{material_id}/memory?target_lang=vi
    PUT    /api/material/{material_id}/memory       (upsert any subset)
    DELETE /api/material/{material_id}/memory       (start over)
    GET    /api/material/{material_id}/memory/briefs?limit=5

The PUT accepts partial bodies — `null` per field leaves the card
intact; `[]` / `{}` explicitly clears. The route does NOT trigger
translation; that's the agent stage's job (M3).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from typoon.api.deps import get_store, require_user
from typoon.api.routes._shared import require_material
from typoon.storage import Store


router = APIRouter(
    prefix="/api/material", tags=["memory"],
    dependencies=[Depends(require_user)],
)


# ── Read ──────────────────────────────────────────────────────────────


class MemoryOut(BaseModel):
    material_id:     int
    source_lang:     str
    target_lang:     str
    characters:      list
    world:           dict
    style:           dict
    glossary:        list
    style_refs:      list
    last_chapter_id: int | None
    updated_at:      str | None


@router.get("/{material_id}/memory", response_model=MemoryOut | None)
async def get_memory(
    material_id: int,
    target_lang: str   = Query(..., min_length=2, max_length=8),
    user:        dict  = Depends(require_user),
    db:          Store = Depends(get_store),
):
    """Return the user's memory row for this material+lang pair, or
    `null` when the user has not started translating yet."""
    await require_material(material_id, db)
    row = await db.get_translator_memory(
        user_id=user["id"], material_id=material_id, target_lang=target_lang,
    )
    if row is None:
        return None
    return MemoryOut(
        material_id=row["material_id"],
        source_lang=row["source_lang"],
        target_lang=row["target_lang"],
        characters=row["characters"],
        world=row["world"],
        style=row["style"],
        glossary=row["glossary"],
        style_refs=row["style_refs"],
        last_chapter_id=row.get("last_chapter_id"),
        updated_at=row.get("updated_at"),
    )


# ── Upsert ────────────────────────────────────────────────────────────


class UpsertMemoryBody(BaseModel):
    # Required on first write (initial create). For subsequent updates
    # the route reads the stored value, so client can omit. We keep it
    # explicit instead of inferring from material.languages because
    # one material can host multiple source langs (raw KR + raw JP).
    source_lang:  str | None = None
    target_lang:  str

    # Per-card partial update. `None` = keep current value. Pass the
    # empty container ([] / {}) to explicitly clear.
    characters:   list | None = None
    world:        dict | None = None
    style:        dict | None = None
    glossary:     list | None = None
    style_refs:   list | None = None


@router.put("/{material_id}/memory", response_model=MemoryOut)
async def upsert_memory(
    material_id: int,
    body:        UpsertMemoryBody,
    user:        dict  = Depends(require_user),
    db:          Store = Depends(get_store),
):
    await require_material(material_id, db)
    existing = await db.get_translator_memory(
        user_id=user["id"], material_id=material_id,
        target_lang=body.target_lang,
    )
    if existing is None and body.source_lang is None:
        raise HTTPException(
            400,
            "source_lang is required when creating memory for the first time",
        )
    source_lang = body.source_lang or existing["source_lang"]  # type: ignore[index]
    row = await db.upsert_translator_memory(
        user_id=user["id"], material_id=material_id,
        source_lang=source_lang, target_lang=body.target_lang,
        characters=body.characters,
        world=body.world,
        style=body.style,
        glossary=body.glossary,
        style_refs=body.style_refs,
    )
    return MemoryOut(
        material_id=row["material_id"],
        source_lang=row["source_lang"],
        target_lang=row["target_lang"],
        characters=row["characters"],
        world=row["world"],
        style=row["style"],
        glossary=row["glossary"],
        style_refs=row["style_refs"],
        last_chapter_id=row.get("last_chapter_id"),
        updated_at=row.get("updated_at"),
    )


# ── Reset ─────────────────────────────────────────────────────────────


@router.delete("/{material_id}/memory", status_code=204)
async def delete_memory(
    material_id: int,
    target_lang: str   = Query(..., min_length=2, max_length=8),
    user:        dict  = Depends(require_user),
    db:          Store = Depends(get_store),
):
    """'Bắt đầu lại' — drop the memory row + its briefs (cascade).
    Idempotent: no row to delete still returns 204."""
    await require_material(material_id, db)
    await db.delete_translator_memory(
        user_id=user["id"], material_id=material_id, target_lang=target_lang,
    )


# ── Briefs (sliding window) ───────────────────────────────────────────


class MemoryBriefOut(BaseModel):
    chapter_id:  int
    position:    int
    number:      str
    label:       str | None
    summary:     str | None
    brief_json:  dict
    created_at:  str | None
    updated_at:  str | None


@router.get(
    "/{material_id}/memory/briefs",
    response_model=list[MemoryBriefOut],
)
async def list_memory_briefs(
    material_id:       int,
    target_lang:       str   = Query(..., min_length=2, max_length=8),
    before_chapter_id: int | None = Query(None),
    limit:             int   = Query(5, ge=1, le=50),
    user:              dict  = Depends(require_user),
    db:                Store = Depends(get_store),
):
    """Sliding window of accumulated briefs. The translate stage pulls
    this with `limit=5` and `before_chapter_id=<current>` before
    spawning to feed the agent the most-recent context.

    Returns empty list when the user has no memory row yet."""
    await require_material(material_id, db)
    mem = await db.get_translator_memory(
        user_id=user["id"], material_id=material_id, target_lang=target_lang,
    )
    if mem is None:
        return []
    rows = await db.list_recent_memory_briefs(
        memory_id=mem["id"], before_chapter_id=before_chapter_id, limit=limit,
    )
    return [MemoryBriefOut(**r) for r in rows]
