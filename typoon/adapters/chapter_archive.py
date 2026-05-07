"""Chapter archive helpers — deterministic Bunle keys.

One canonical key per chapter for prepared and render archives. Stage
overwrites in place; readers gate on `chapters.rendered`, so no mid-write
race exists in the queue worker model.

URL ↔ archive bijection (cache-bust, mid-stream consistency for the web
viewer) is an app-layer concern handled separately when needed.
"""

from __future__ import annotations

from pathlib import Path

import bunle

from typoon.adapters.artifact_store import ArtifactStore


def archive_dir_prefix(project_id: int, chapter_id: int) -> str:
    return f"p/{project_id}/c/{chapter_id}"


def prepared_key(project_id: int, chapter_id: int) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/prepared.bnl"


def render_key(project_id: int, chapter_id: int) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/render.bnl"


def masks_key(project_id: int, chapter_id: int) -> str:
    return f"{archive_dir_prefix(project_id, chapter_id)}/masks.npz"


async def pack_and_upload(
    *,
    src_dir: Path,
    archive_path: Path,
    key: str,
    store: ArtifactStore,
) -> int:
    """Pack `src_dir` into a Bunle archive at `archive_path`, validate, upload.
    Returns the archive page count.
    """
    bunle.pack_dir(str(src_dir), str(archive_path))
    bunle.validate(str(archive_path))
    info = bunle.info(str(archive_path))
    await store.put_file(key, archive_path)
    return info["page_count"]
