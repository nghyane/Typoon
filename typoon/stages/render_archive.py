"""Render orchestrator — translate.Chapter → render.bnl on store.

Wraps `stages.render.render_chapter`: produces WebP pages in tmp, packs
them into a Bunle archive, uploads. The CPU-heavy render runs on a
worker thread so the event loop stays responsive while the upload awaits.

The persistent `chapters.rendered` flag flip is the caller's
responsibility (`Store.set_rendered(True)` after this returns).
Concurrency comes from `Store.claim_task('render', ...)` — only one
worker can hold the render slot at a time.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import pack_and_upload, render_key
from typoon.adapters.mask_store import MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain import translate
from typoon.domain.scan import PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook
from typoon.stages._workdir import workdir
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
    archive_salt: bytes,
    hook: Hook | None = None,
    artifacts: ArtifactSink | None = None,
    work: Path | None = None,
    skip_pages: frozenset[int] = frozenset(),
) -> tuple[str, int]:
    """Render translation, pack `render.bnl`, upload.

    Returns (locator, public_page_count). public_page_count excludes
    pages dropped via `skip_pages` so the caller can persist the actual
    number of pages the reader will see.
    """
    with workdir(work) as tmp:
        out_dir = tmp / "render_webp"
        archive_path = tmp / "render.bnl"

        rendered = await asyncio.to_thread(
            render_chapter,
            translated, out_dir, reader, runtime, page_geoms, masks,
            chapter_id=chapter_id, project_id=project_id,
            hook=hook, artifacts=artifacts, skip_pages=skip_pages,
        )

        page_count, locator = await pack_and_upload(
            src_dir=out_dir,
            archive_path=archive_path,
            key=render_key(project_id, chapter_id, archive_salt),
            store=store,
        )
        # `rendered.pages` is the same set the bunle was packed from,
        # so its length must match the archive's page_count. We return
        # the bunle count as the source of truth.
        _ = rendered  # (returned only for caller introspection if desired)
        return locator, page_count
