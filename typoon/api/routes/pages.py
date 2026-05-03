"""Rendered page serving."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from typoon.api.deps import get_paths, get_store
from typoon.paths import Paths, ProjectPaths
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["pages"])


async def _chapter_paths(project_id: int, chapter_id: int, db: Store, paths: Paths):
    proj = await db.get_project(project_id)
    if proj is None:
        raise HTTPException(404, "Project not found")
    return ProjectPaths(paths.projects, proj["slug"]).chapter(chapter_id)


@router.get("/{project_id}/chapters/{chapter_id}/pages/{index}")
async def get_page(
    project_id: int,
    chapter_id: int,
    index:      int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    cp   = await _chapter_paths(project_id, chapter_id, db, paths)
    path = cp.rendered(index)
    if not path.exists():
        raise HTTPException(404, "Page not rendered yet")
    return FileResponse(str(path), media_type="image/png")
