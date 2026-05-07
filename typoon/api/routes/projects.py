"""Projects and chapters routes."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.projects import Projects
from typoon.api.deps import get_artifact_store, get_paths, get_store
from typoon.api.models import ChapterOut, ProjectOut, Progress
from typoon.paths import Paths, ProjectPaths
from typoon.runs.events import Hook
from typoon.storage import Store

router = APIRouter(prefix="/api/projects", tags=["projects"])


# ── Request bodies ────────────────────────────────────────────────────


class ImportBody(BaseModel):
    folder:      str
    title:       str
    source_lang: str = "ko"
    target_lang: str = "vi"


# ── Helpers ───────────────────────────────────────────────────────────


async def _require_project(project_id: int, db: Store) -> dict:
    proj = await db.get_project(project_id)
    if proj is None:
        raise HTTPException(404, "Project not found")
    return proj


def _chapter_out(data: dict) -> ChapterOut:
    page_count = int(data.get("page_count") or 0)
    progress_data = data.get("progress")
    progress = (
        Progress(
            stage=progress_data.get("stage") or data.get("stage") or "",
            page_index=int(progress_data.get("page_index") or 0),
            page_total=int(progress_data.get("page_total") or page_count),
        )
        if progress_data and data["state"] == "running"
        else None
    )
    return ChapterOut(
        chapter_id=data["chapter_id"],
        project_id=data["project_id"],
        idx=data["idx"],
        title=data.get("title"),
        state=data["state"],
        stage=data.get("stage") or "",
        page_count=page_count,
        error=data.get("error") or "",
        updated_at=data.get("updated_at"),
        progress=progress,
    )


# ── Projects ──────────────────────────────────────────────────────────


@router.get("", response_model=list[ProjectOut])
async def list_projects(db: Store = Depends(get_store)):
    return [ProjectOut.from_row(p) for p in await db.list_projects()]


@router.post("", response_model=ProjectOut, status_code=201)
async def import_project(
    body:  ImportBody,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
):
    folder = Path(body.folder)
    if not folder.is_dir():
        raise HTTPException(400, f"Not a directory: {body.folder}")
    project_id = await Projects(db, paths, store).import_new(
        folder, body.title, body.source_lang, body.target_lang, Hook()
    )
    proj = await db.get_project(project_id)
    return ProjectOut.from_row(proj)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(project_id: int, db: Store = Depends(get_store)):
    proj = await db.get_project(project_id)
    return ProjectOut.from_row(proj)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: int,
    db:    Store = Depends(get_store),
    paths: Paths = Depends(get_paths),
):
    proj = await _require_project(project_id, db)
    await db.delete_project(project_id)
    shutil.rmtree(ProjectPaths(paths.projects, proj["slug"]).root, ignore_errors=True)
    shutil.rmtree(paths.artifacts / "p" / str(project_id), ignore_errors=True)


# ── Chapters ──────────────────────────────────────────────────────────


@router.get("/{project_id}/chapters", response_model=list[ChapterOut])
async def list_chapters(
    project_id: int,
    db:    Store = Depends(get_store),
):
    await _require_project(project_id, db)
    rows = await db.get_chapters_with_status(project_id)
    return [_chapter_out(r) for r in rows]


@router.get("/{project_id}/chapters/{chapter_id}", response_model=ChapterOut)
async def get_chapter(
    project_id: int,
    chapter_id: int,
    db:    Store = Depends(get_store),
):
    await _require_project(project_id, db)
    data = await db.get_chapter_with_status(chapter_id, project_id)
    if data is None:
        raise HTTPException(404, "Chapter not found")
    return _chapter_out(data)


@router.post("/{project_id}/chapters/{chapter_id}/redo", response_model=ChapterOut)
async def redo_chapter(
    project_id: int,
    chapter_id: int,
    db:    Store         = Depends(get_store),
    paths: Paths         = Depends(get_paths),
    store: ArtifactStore = Depends(get_artifact_store),
):
    proj = await _require_project(project_id, db)
    ch   = await db.get_chapter(chapter_id)
    if ch is None or ch["project_id"] != project_id:
        raise HTTPException(404, "Chapter not found")
    await Projects(db, paths, store).redo(proj["slug"], [ch["idx"]])
    data = await db.get_chapter_with_status(chapter_id, project_id)
    return _chapter_out(data)
