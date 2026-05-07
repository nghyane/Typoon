"""Project + chapter CRUD routes."""

from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.projects import Projects
from typoon.api.deps import get_artifact_store, get_paths, get_store
from typoon.api.models import ChapterOut, ProjectOut
from typoon.api.routes._shared import (
    chapter_out, require_chapter, require_project,
)
from typoon.paths import Paths, ProjectPaths, slugify
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["projects"])


# ── Request bodies ────────────────────────────────────────────────────


class CreateProjectBody(BaseModel):
    title:       str
    description: str | None = None
    source_lang: str        = "en"
    target_lang: str        = "vi"


class SettingsBody(BaseModel):
    target_lang: str | None = None
    settings:    dict | None = None  # arbitrary JSON overrides
    title:       str | None = None
    description: str | None = None


# ── Projects ──────────────────────────────────────────────────────────


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: Store = Depends(get_store)):
    return [ProjectOut.from_row(p) for p in await db.list_projects()]


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body:  CreateProjectBody,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    """Create an empty project. Cover, chapters etc come later via dedicated
    endpoints (POST /{id}/cover, POST /{id}/pull)."""
    title = body.title.strip()
    if not title:
        raise HTTPException(400, "title required")

    slug = slugify(title)
    pid  = await db.get_or_create_project(
        slug=slug, title=title,
        source_lang=body.source_lang, target_lang=body.target_lang,
    )
    if body.description:
        await db.update_project_metadata(pid, description=body.description)
    ProjectPaths(paths.projects, slug).ensure()

    proj = await db.get_project(pid)
    return ProjectOut.from_row(proj)


@router.post("/{project_id}/cover", response_model=ProjectOut)
async def upload_cover(
    project_id: int,
    file:       UploadFile = File(...),
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    """Upload a cover image. Re-encoded to JPEG q=88 under projects/<slug>/cover.jpg."""
    proj = await require_project(project_id, db)
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "cover must be an image")

    raw = await file.read()
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image: {e}") from e

    proj_paths = ProjectPaths(paths.projects, proj["slug"])
    proj_paths.ensure()
    img.save(proj_paths.cover, format="JPEG", quality=88)
    await db.update_project_metadata(project_id, cover_path=str(proj_paths.cover))

    proj = await db.get_project(project_id)
    return ProjectOut.from_row(proj)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: int, db: Store = Depends(get_store)):
    proj = await require_project(project_id, db)
    return ProjectOut.from_row(proj)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    proj = await require_project(project_id, db)
    await db.delete_project(project_id)
    shutil.rmtree(ProjectPaths(paths.projects, proj["slug"]).root, ignore_errors=True)
    shutil.rmtree(paths.artifacts / "p" / str(project_id), ignore_errors=True)


# ── Settings ──────────────────────────────────────────────────────────


@router.get("/{project_id}/settings")
async def get_settings(project_id: int, db: Store = Depends(get_store)):
    proj = await require_project(project_id, db)
    return {
        "project_id":  project_id,
        "target_lang": proj["target_lang"],
        "title":       proj["title"],
        "description": proj.get("description"),
        "settings":    await db.get_project_settings(project_id),
    }


@router.patch("/{project_id}/settings")
async def patch_settings(
    project_id: int,
    body:       SettingsBody,
    db:         Store = Depends(get_store),
):
    await require_project(project_id, db)
    await db.update_project_metadata(
        project_id,
        title=body.title,
        description=body.description,
        target_lang=body.target_lang,
        settings=body.settings,
    )
    proj = await db.get_project(project_id)
    return {
        "project_id":  project_id,
        "target_lang": proj["target_lang"],
        "title":       proj["title"],
        "description": proj.get("description"),
        "settings":    await db.get_project_settings(project_id),
    }


# ── Chapters ──────────────────────────────────────────────────────────


@router.get("/{project_id}/chapters", response_model=list[ChapterOut])
async def list_chapters(
    project_id: int,
    db: Store = Depends(get_store),
):
    await require_project(project_id, db)
    rows = await db.get_chapters_with_status(project_id)
    return [chapter_out(r) for r in rows]


@router.get("/{project_id}/chapters/{chapter_id}", response_model=ChapterOut)
async def get_chapter(
    project_id: int,
    chapter_id: int,
    db: Store = Depends(get_store),
):
    await require_project(project_id, db)
    data = await db.get_chapter_with_status(chapter_id, project_id)
    if data is None:
        raise HTTPException(404, "Chapter not found")
    return chapter_out(data)


@router.delete("/{project_id}/chapters/{chapter_id}", status_code=204)
async def delete_chapter(
    project_id: int,
    chapter_id: int,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
):
    await require_chapter(project_id, chapter_id, db)
    # Drop chapter row (cascades) — caller wanted full removal, not just
    # derived data.
    deleted = await db.delete_chapter(chapter_id)
    if not deleted:
        raise HTTPException(404, "Chapter not found")
    # Best-effort artifact cleanup. The chapter row is gone from DB so
    # missing keys are fine.
    from typoon.adapters.chapter_archive import (
        prepared_key, render_key, masks_key,
    )
    for key in (
        prepared_key(project_id, chapter_id),
        render_key(project_id, chapter_id),
        masks_key(project_id, chapter_id),
    ):
        await store.delete(key)


@router.post("/{project_id}/chapters/{chapter_id}/redo", response_model=ChapterOut)
async def redo_chapter(
    project_id: int,
    chapter_id: int,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
):
    proj = await require_project(project_id, db)
    ch   = await require_chapter(project_id, chapter_id, db)
    await Projects(db, paths, store).redo(proj["slug"], [ch["idx"]])
    data = await db.get_chapter_with_status(chapter_id, project_id)
    return chapter_out(data)


# ── Chapter brief ─────────────────────────────────────────────────────


@router.get("/{project_id}/chapters/{chapter_id}/brief")
async def get_brief(
    project_id: int,
    chapter_id: int,
    db: Store = Depends(get_store),
):
    await require_chapter(project_id, chapter_id, db)
    brief = await db.get_chapter_brief(chapter_id)
    if brief is None:
        raise HTTPException(404, "Brief not generated yet")
    return brief
