"""Rendered page + PDF serving."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from typoon.api.deps import get_paths, get_store
from fastapi import Depends
from typoon.paths import Paths, ProjectPaths
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["pages"])


async def _cp(slug: str, chapter_id: int, db: Store, paths: Paths):
    rows = await db.list_projects()
    proj = next((r for r in rows if r["slug"] == slug), None)
    if proj is None:
        raise HTTPException(404, "Project not found")
    return ProjectPaths(paths.projects, slug).chapter(chapter_id)


@router.get("/{slug}/chapters/{chapter_id}/pages/{index}")
async def get_page(
    slug:       str,
    chapter_id: int,
    index:      int,
    db:    Store = Depends(get_store),
    paths: Paths  = Depends(get_paths),
):
    cp   = await _cp(slug, chapter_id, db, paths)
    path = cp.rendered(index)
    if not path.exists():
        raise HTTPException(404, "Page not rendered yet")
    return FileResponse(str(path), media_type="image/png")


@router.get("/{slug}/chapters/{chapter_id}/pdf")
async def export_pdf(
    slug:       str,
    chapter_id: int,
    db:    Store = Depends(get_store),
    paths: Paths  = Depends(get_paths),
):
    cp      = await _cp(slug, chapter_id, db, paths)
    renders = sorted(cp.render.glob("*.png")) if cp.is_rendered else []
    if not renders:
        raise HTTPException(404, "No rendered pages")

    from PIL import Image
    import io
    images = [Image.open(p).convert("RGB") for p in renders]
    buf = io.BytesIO()
    images[0].save(buf, format="PDF", save_all=True, append_images=images[1:])
    buf.seek(0)
    filename = f"{slug}-ch{chapter_id}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
