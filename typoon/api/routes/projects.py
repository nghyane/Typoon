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
    from_ch: float = 0
    to_ch:   float = 0


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
    indices = None
    if body.from_ch > 0 or body.to_ch > 0:
        projects = await db.list_projects()
        proj = next((p for p in projects if p["slug"] == slug), None)
        if proj is None:
            raise HTTPException(404, "Project not found")
        all_chs = await db.get_all_chapters(proj["id"])
        lo = body.from_ch or all_chs[0]["idx"]
        hi = body.to_ch   or all_chs[-1]["idx"]
        indices = [c["idx"] for c in all_chs if lo <= c["idx"] <= hi]
    count = await Projects(db, paths).redo(slug, indices)
    return {"reset": count}
