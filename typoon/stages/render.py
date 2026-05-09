"""Render stage — TranslatedChapter + masks → RenderedChapter on disk.

Receives pre-loaded geometry and an in-memory MaskStore. Source pixels
come from a PreparedReader. Render output is written as WebP-lossless
files into `out_dir`; the orchestrator (render_archive) packs them into
a Bunle archive and uploads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.prepared_reader import PreparedReader
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain import render, translate
from typoon.domain.scan import PageGeometry
from typoon.runs.artifacts import ArtifactSink
from typoon.runs.events import Hook, PageDone


def render_chapter(
    translated: translate.Chapter,
    out_dir: Path,
    reader: PreparedReader,
    runtime: VisionRuntime,
    page_geoms: dict[int, PageGeometry],
    masks: MaskStore,
    *,
    chapter_id: int = 0,
    project_id: int = 0,
    hook: Hook | None = None,
    artifacts: ArtifactSink | None = None,
    skip_pages: frozenset[int] = frozenset(),
) -> render.Chapter:
    """Erase source text, render translations, write WebP pages into out_dir.

    page_geoms: pre-loaded from load_translated_with_geometry.
    masks:      in-memory mask store (typically loaded from masks.npz).
    skip_pages: page indices that are entirely non-diegetic — drop from
                the output archive so they never reach the reader.
                Output WebPs are renumbered contiguously from 0 to
                preserve a gapless reading experience.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    import typoon_render

    rendered_pages = []
    out_index = 0

    for tp in translated.pages:
        if tp.index in skip_pages:
            if artifacts is not None:
                # Record the drop so visual verification can confirm
                # which pages were excluded by the brief.
                artifacts.write_image(
                    "06_render", f"{tp.index:04d}_dropped.png", reader.read_rgb(tp.index)
                )
            if hook is not None:
                hook.on(PageDone(chapter_id=chapter_id, project_id=project_id, stage="render",
                                 page_index=tp.index, page_total=len(translated.pages)))
            continue

        original   = reader.read_rgb(tp.index)
        canvas     = _to_rgba(original)
        page_masks = masks.page_masks(tp.index)

        erase_masks = [
            m
            for tb in tp.bubbles
            for bm in [page_masks.get(tb.idx)]
            if bm is not None
            for m in bm.erase_masks
        ]
        if erase_masks and runtime.eraser is not None:
            runtime.eraser.erase(canvas, erase_masks)

        clean = canvas[:, :, :3]

        pg_geom     = page_geoms.get(tp.index)
        geom_by_idx = {bg.bubble_idx: bg for bg in pg_geom.bubbles} if pg_geom else {}

        active_triples = [
            (tb, geom_by_idx[tb.idx].polygon, tb.translated_text)
            for tb in tp.bubbles
            if tb.kind != "skip"
            and tb.translated_text.strip()
            and tb.idx in geom_by_idx
        ]
        active   = [t[0] for t in active_triples]
        polygons = [t[1] for t in active_triples]
        texts    = [t[2] for t in active_triples]

        result = typoon_render.typoon_render.render(
            original, clean, polygons, texts, original.shape[1]
        )

        active_info = {tb.idx: rb for tb, rb in zip(active, result.bubbles)}
        rendered_bubbles = tuple(
            render.Bubble(
                source=tb,
                font_size=active_info[tb.idx].font_size_px if tb.idx in active_info else 0,
                overflow=active_info[tb.idx].overflow if tb.idx in active_info else False,
            )
            for tb in tp.bubbles
        )

        _write_webp(out_dir / f"{out_index:04d}.webp", result.image)
        out_index += 1

        if hook is not None:
            hook.on(PageDone(chapter_id=chapter_id, project_id=project_id, stage="render",
                             page_index=tp.index, page_total=len(translated.pages)))

        if artifacts is not None:
            artifacts.write_image("06_render", f"{tp.index:04d}_rendered.png", result.image)

        rendered_pages.append(render.Page(source=tp, bubbles=rendered_bubbles))

    return render.Chapter(source=translated, pages=tuple(rendered_pages))


def _to_rgba(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.dstack([image, np.full((h, w), 255, dtype=np.uint8)])


def _write_webp(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(
        path, format="WEBP", lossless=True, quality=100, method=6,
    )
