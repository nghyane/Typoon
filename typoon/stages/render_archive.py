"""Render orchestrator — translate.Chapter → render.bnl on store.

Wraps `stages.render.render_chapter`: produces WebP pages in tmp, packs
them into a Bunle archive, uploads to the chapter's render key. The CAS
state-machine flip (`claim_render_job` / `finish_render_job`) is the
caller's responsibility.

The CPU-heavy `render_chapter` step is offloaded to a worker thread so
the event loop stays responsive while the upload awaits.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import pack_and_upload, render_key
from typoon.adapters.mask_store import MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain import render as render_dom
from typoon.domain import translate
from typoon.domain.scan import PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook
from typoon.stages.render import render_chapter


async def render_chapter_to_archive(
    translated: translate.Chapter,
    *,
    project_id: int,
    chapter_id: int,
    reader: PreparedReader,
    runtime: VisionRuntime,
    page_geoms: dict[int, PageGeometry],
    masks: MaskStore,
    store: ArtifactStore,
    hook: Hook | None = None,
    artifacts: ArtifactSink | None = None,
    workdir: Path | None = None,
) -> tuple[str, int, render_dom.Chapter]:
    """Render translation, pack the render archive, upload. Returns
    (render_key, page_count, render.Chapter).
    """
    workdir_ctx = (
        _NullCtx(workdir) if workdir else tempfile.TemporaryDirectory()
    )
    with workdir_ctx as tmp_str:
        tmp = Path(tmp_str)
        out_dir = tmp / "render_webp"
        archive_path = tmp / "render.bnl"

        rendered = await asyncio.to_thread(
            render_chapter,
            translated, out_dir, reader, runtime, page_geoms, masks,
            chapter_id=chapter_id, project_id=project_id,
            hook=hook, artifacts=artifacts,
        )

        key = render_key(project_id, chapter_id)
        page_count = await pack_and_upload(
            src_dir=out_dir,
            archive_path=archive_path,
            key=key,
            store=store,
        )
        return key, page_count, rendered


class _NullCtx:
    def __init__(self, path: Path) -> None:
        self._path = str(path)

    def __enter__(self) -> str:
        Path(self._path).mkdir(parents=True, exist_ok=True)
        return self._path

    def __exit__(self, *_) -> None:
        pass
