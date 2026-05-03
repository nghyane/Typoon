"""Projects + chapters routes."""

from __future__ import annotations

import shutil
from pathlib import Path

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


@router.get("")
async def list_projects(
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    return await Projects(db, paths).get_status()


@router.post("", status_code=201)
async def import_project(
    body:  ImportBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    folder = Path(body.folder)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {body.folder}")
    slug = await Projects(db, paths).import_new(
        folder, body.title, body.source_lang, body.target_lang, Hook()
    )
    return {"slug": slug}


@router.delete("/{slug}", status_code=204)
async def delete_project(
    slug:  str,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    projects = await db.list_projects()
    proj = next((p for p in projects if p["slug"] == slug), None)
    if proj is None:
        raise HTTPException(404, "Project not found")
    await db.delete_project(proj["id"])
    shutil.rmtree(ProjectPaths(paths.projects, slug).root, ignore_errors=True)


@router.post("/{slug}/redo")
async def redo_project(
    slug:  str,
    body:  RedoBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    all_projects = await db.list_projects()
    proj = next((p for p in all_projects if p["slug"] == slug), None)
    if proj is None:
        raise HTTPException(404, "Project not found")

    indices = None
    if body.chapter_ids:
        all_chs   = await db.get_all_chapters(proj["id"])
        idx_by_id = {c["id"]: c["idx"] for c in all_chs}
        unknown   = [cid for cid in body.chapter_ids if cid not in idx_by_id]
        if unknown:
            raise HTTPException(400, f"Unknown chapter ids: {unknown}")
        indices = [idx_by_id[cid] for cid in body.chapter_ids]

    count = await Projects(db, paths).redo(slug, indices)
    return {"reset": count}
