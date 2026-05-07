"""Glossary CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.api.deps import get_store
from typoon.api.models import GlossaryTermOut
from typoon.api.routes._shared import require_project
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["glossary"])


class GlossaryBody(BaseModel):
    source_term: str
    target_term: str
    notes:       str | None = None


@router.get("/{project_id}/glossary", response_model=list[GlossaryTermOut])
async def list_terms(project_id: int, db: Store = Depends(get_store)):
    await require_project(project_id, db)
    rows = await db.list_glossary(project_id)
    return [GlossaryTermOut(**r) for r in rows]


@router.post(
    "/{project_id}/glossary",
    response_model=GlossaryTermOut,
    status_code=201,
)
async def create_term(
    project_id: int,
    body:       GlossaryBody,
    db:         Store = Depends(get_store),
):
    await require_project(project_id, db)
    if not body.source_term.strip():
        raise HTTPException(400, "source_term required")
    term_id = await db.upsert_glossary_term(
        project_id, body.source_term.strip(), body.target_term.strip(), body.notes,
    )
    return GlossaryTermOut(
        id=term_id,
        source_term=body.source_term.strip(),
        target_term=body.target_term.strip(),
        notes=body.notes,
    )


@router.patch(
    "/{project_id}/glossary/{term_id}",
    response_model=GlossaryTermOut,
)
async def update_term(
    project_id: int,
    term_id:    int,
    body:       GlossaryBody,
    db:         Store = Depends(get_store),
):
    await require_project(project_id, db)
    rows = await db.list_glossary(project_id)
    if not any(r["id"] == term_id for r in rows):
        raise HTTPException(404, "Term not found")
    # upsert by source_term replaces target/notes; for true rename we'd
    # delete + insert. Treat the existing term_id as authoritative.
    await db.upsert_glossary_term(
        project_id, body.source_term.strip(), body.target_term.strip(), body.notes,
    )
    return GlossaryTermOut(
        id=term_id,
        source_term=body.source_term.strip(),
        target_term=body.target_term.strip(),
        notes=body.notes,
    )


@router.delete(
    "/{project_id}/glossary/{term_id}",
    status_code=204,
)
async def delete_term(
    project_id: int,
    term_id:    int,
    db:         Store = Depends(get_store),
):
    await require_project(project_id, db)
    if not await db.delete_glossary_term(project_id, term_id):
        raise HTTPException(404, "Term not found")
