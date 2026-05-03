"""Projects + chapters routes."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.projects import Projects
from typoon.api.deps import get_paths, get_store
from typoon.paths import Paths, ProjectPaths
from typoon.runs.events import Hook
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ImportBody(BaseModel):
    folder:      str
    title:       str
    source_lang: str = "ko"
    target_lang: str = "vi"


class RedoBody(BaseModel):
    chapter_ids: list[int] = []  # empty = redo all


async def _require_project(project_id: int, db: Store) -> dict:
    proj = await db.get_project(project_id)
    if proj is None:
        raise HTTPException(404, "Project not found")
    return proj


@router.get("/{project_id}/chapters")
async def list_chapters(
    project_id: int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    proj = await _require_project(project_id, db)
    return await db.get_chapters_with_status(project_id, paths.projects, proj["slug"])


@router.get("")
async def list_projects(db: Store = Depends(get_store)):
    return await db.list_projects()


@router.post("", status_code=201)
async def import_project(
    body:  ImportBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    from pathlib import Path
    folder = Path(body.folder)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {body.folder}")
    slug = await Projects(db, paths).import_new(
        folder, body.title, body.source_lang, body.target_lang, Hook()
    )
    proj = await db.list_projects()
    return next(p for p in proj if p["slug"] == slug)


@router.get("/{project_id}")
async def get_project(
    project_id: int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    proj = await _require_project(project_id, db)
    status = await Projects(db, paths).get_status(proj["slug"])
    return status[0] if status else proj


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    proj = await _require_project(project_id, db)
    await db.delete_project(project_id)
    shutil.rmtree(ProjectPaths(paths.projects, proj["slug"]).root, ignore_errors=True)


@router.post("/{project_id}/redo")
async def redo_project(
    project_id: int,
    body:       RedoBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    proj = await _require_project(project_id, db)

    indices = None
    if body.chapter_ids:
        all_chs   = await db.get_all_chapters(project_id)
        idx_by_id = {c["id"]: c["idx"] for c in all_chs}
        unknown   = [cid for cid in body.chapter_ids if cid not in idx_by_id]
        if unknown:
            raise HTTPException(400, f"Unknown chapter ids: {unknown}")
        indices = [idx_by_id[cid] for cid in body.chapter_ids]

    count = await Projects(db, paths).redo(proj["slug"], indices)
    return {"reset": count}
