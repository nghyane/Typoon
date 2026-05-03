"""Render stage — TranslatedChapter → RenderedChapter.

Receives pre-loaded geometry — caller must pass page_geoms from
adapters.loader.load_translated_with_geometry to avoid double scan.npz read.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain import render, translate
from typoon.domain.scan import PageGeometry
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink


def render_chapter(
    translated: translate.Chapter,
    cp: ChapterPaths,
    runtime: VisionRuntime,
    page_geoms: dict[int, PageGeometry],
    *,
    artifacts: ArtifactSink | None = None,
) -> render.Chapter:
    """Erase source text, render translations, write PNGs to cp.render/.

    page_geoms: pre-loaded from load_translated_with_geometry — not re-read here.
    Masks loaded per-page from cp.masks/ — one file open per page.
    """
    cp.render.mkdir(parents=True, exist_ok=True)

    import typoon_render

    rendered_pages = []

    for tp in translated.pages:
        original   = _load_rgb(translated.scan.prepared.page_path(tp.index))
        canvas     = _to_rgba(original)
        page_masks = MaskStore.load_page(cp, tp.index)

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

        # Build aligned (active, polygons, texts) in one pass — no reassign
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

        image_path = cp.rendered(tp.index)
        _write_rgb(image_path, result.image)

        if artifacts is not None:
            artifacts.write_image("06_render", f"{tp.index:04d}_rendered.png", result.image)

        rendered_pages.append(render.Page(
            source=tp, bubbles=rendered_bubbles, image_path=image_path,
        ))

    return render.Chapter(source=translated, pages=tuple(rendered_pages))


def _load_rgb(path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read prepared page: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_rgba(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.dstack([image, np.full((h, w), 255, dtype=np.uint8)])


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write rendered page: {path}")
