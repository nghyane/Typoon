"""Rendered page + PDF serving."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

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


@router.get("/{project_id}/chapters/{chapter_id}/pdf")
async def export_pdf(
    project_id: int,
    chapter_id: int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    cp      = await _chapter_paths(project_id, chapter_id, db, paths)
    renders = sorted(cp.render.glob("*.png")) if cp.is_rendered else []
    if not renders:
        raise HTTPException(404, "No rendered pages")

    from PIL import Image
    import io
    images = [Image.open(p).convert("RGB") for p in renders]
    buf    = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="chapter-{chapter_id}.pdf"'},
    )
