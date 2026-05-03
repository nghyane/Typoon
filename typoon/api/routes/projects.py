"""Projects + chapters routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.projects import Projects, ChapterStatus, ProjectStatus
from typoon.api.deps import get_paths, get_store
from typoon.paths import Paths
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ImportBody(BaseModel):
    folder: str
    title: str
    source_lang: str = "ko"
    target_lang: str = "vi"


class RedoBody(BaseModel):
    from_ch: float = 0
    to_ch:   float = 0


async def _projects(db: Store, paths: Paths) -> Projects:
    return Projects(db, paths)


@router.get("")
async def list_projects(
    db:    Store = Depends(get_store),
    paths: Paths  = Depends(get_paths),
):
    p = await _projects(db, paths)
    return await p.get_status()


@router.post("", status_code=201)
async def import_project(
    body:  ImportBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    folder = Path(body.folder)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {body.folder}")
    p = await _projects(db, paths)
    from typoon.runs.events import Hook
    slug = await p.import_new(folder, body.title, body.source_lang, body.target_lang, Hook())
    return {"slug": slug}


@router.delete("/{slug}", status_code=204)
async def delete_project(
    slug:  str,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    rows = await db.list_projects()
    proj = next((r for r in rows if r["slug"] == slug), None)
    if proj is None:
        raise HTTPException(404, "Project not found")
    import shutil
    from typoon.paths import ProjectPaths
    await db._db.execute("DELETE FROM projects WHERE slug=?", (slug,))
    await db._db.commit()
    pp = ProjectPaths(paths.projects, slug)
    shutil.rmtree(pp.root, ignore_errors=True)


@router.post("/{slug}/redo")
async def redo_project(
    slug:  str,
    body:  RedoBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    p = await _projects(db, paths)
    indices = None
    if body.from_ch > 0 or body.to_ch > 0:
        proj = next((r for r in await db.list_projects() if r["slug"] == slug), None)
        if proj is None:
            raise HTTPException(404, "Project not found")
        all_chs = await db.get_all_chapters(proj["id"])
        lo = body.from_ch or all_chs[0]["idx"]
        hi = body.to_ch   or all_chs[-1]["idx"]
        indices = [c["idx"] for c in all_chs if lo <= c["idx"] <= hi]
    count = await p.redo(slug, indices)
    return {"reset": count}
