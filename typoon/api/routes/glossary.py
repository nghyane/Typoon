"""Glossary CRUD — per-user, per (source_lang, target_lang).

Replaces the old per-project glossary. A user maintains one glossary
per language pair; translation spawn merges it with community_glossary
(scoped + global) to compute the cache fingerprint.

Community glossary editing is admin-only at this phase; users can only
query the merged result via the translation flow.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from typoon.api.deps import get_store, require_user
from typoon.api.models import GlossaryTermOut
from typoon.storage import Store


router = APIRouter(
    prefix="/api/glossary", tags=["glossary"],
    dependencies=[Depends(require_user)],
)


class GlossaryBody(BaseModel):
    source_lang: str
    target_lang: str
    source_term: str
    target_term: str
    notes:       str | None = None


@router.get("", response_model=list[GlossaryTermOut])
async def list_terms(
    source_lang: str | None = Query(None),
    target_lang: str | None = Query(None),
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """List the caller's glossary, optionally filtered to a single
    language pair (the spawn modal calls this with both langs)."""
    rows = await db.list_user_glossary(
        user["id"], source_lang=source_lang, target_lang=target_lang,
    )
    return [GlossaryTermOut(**r) for r in rows]


@router.post("", response_model=GlossaryTermOut, status_code=201)
async def upsert_term(
    body: GlossaryBody,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    """Idempotent upsert on (user, source_lang, target_lang, source_term).
    A second POST with the same term updates the target/notes."""
    source_term = body.source_term.strip()
    target_term = body.target_term.strip()
    if not source_term or not target_term:
        raise HTTPException(400, "source_term and target_term required")
    term_id = await db.upsert_user_glossary_term(
        user_id=user["id"],
        source_lang=body.source_lang, target_lang=body.target_lang,
        source_term=source_term, target_term=target_term,
        notes=body.notes,
    )
    return GlossaryTermOut(
        id=term_id,
        source_lang=body.source_lang, target_lang=body.target_lang,
        source_term=source_term, target_term=target_term,
        notes=body.notes,
    )


@router.delete("/{term_id}", status_code=204)
async def delete_term(
    term_id: int,
    user: dict  = Depends(require_user),
    db:   Store = Depends(get_store),
):
    if not await db.delete_user_glossary_term(user["id"], term_id):
        raise HTTPException(404, "Term not found")
