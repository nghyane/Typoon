"""Render orchestrator — translate.Chapter → render.bnl on store.

Wraps `stages.render.render_chapter`: produces WebP pages in tmp, packs
them into a Bunle archive, uploads. The CPU-heavy render runs on a
worker thread so the event loop stays responsive while the upload
awaits.

Target shape:
  - `target_kind='draft'`        → `d/{draft_id}/render.bnl`
                                    shared by every translation that
                                    references the draft and has no edits.
  - `target_kind='translation'`  → `t/{translation_id}/render.bnl`
                                    fork the draft render when sparse
                                    edits diverge.

The DB pointer flip
(`Store.update_translation_archive` or update_draft_archive) is the
caller's responsibility. Concurrency comes from
`Store.claim_task('render', ...)` — only one worker holds the slot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from typoon.adapters.artifact_store import ArtifactStore
from typoon.adapters.chapter_archive import pack_and_upload, render_key
from typoon.adapters.mask_store import MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.vision.runtime import VisionRuntime
from typoon.domain import translate
from typoon.domain.scan import PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook
from typoon.stages._workdir import workdir
from typoon.stages.render import render_chapter


RenderTargetKind = Literal["draft", "translation"]


async def render_chapter_to_archive(
    translated: translate.Chapter,
    *,
    target_kind: RenderTargetKind,
    target_id:   int,
    chapter_id:  int,
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
        out_dir = tmp / "render_jpg"
        archive_path = tmp / "render.bnl"

        rendered = await render_chapter(
            translated, out_dir, reader, runtime, page_geoms, masks,
            chapter_id=chapter_id,
            target_kind=target_kind, target_id=target_id,
            hook=hook, artifacts=artifacts, skip_pages=skip_pages,
        )

        page_count, locator = await pack_and_upload(
            src_dir=out_dir,
            archive_path=archive_path,
            key=render_key(target_kind, target_id, archive_salt),
            store=store,
        )
        _ = rendered  # returned only for caller introspection if desired
        return locator, page_count
