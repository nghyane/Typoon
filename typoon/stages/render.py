"""Render stage — TranslatedChapter + geometry + masks → RenderedChapter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from typoon.adapters.mask_store import MaskStore, load_scan_geometry
from typoon.domain.render import Bubble as RenderedBubble, Chapter as RenderedChapter, Page as RenderedPage
from typoon.domain.translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.paths import ChapterPaths
from typoon.runs.artifacts import ArtifactSink


def render_chapter(
    translated: TranslatedChapter,
    cp: ChapterPaths,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> RenderedChapter:
    """Erase source text, render translations, write PNGs to cp.render/.

    Geometry loaded from cp.scan (scan.npz).
    Masks loaded per-page from cp.masks/ — one file open per page.
    """
    cp.render.mkdir(parents=True, exist_ok=True)

    import typoon_render

    # Load geometry once — mmap, no full RAM load
    geometry = {pg.page_index: pg for pg in load_scan_geometry(cp)}

    rendered_pages = []

    for tp in translated.pages:
        original = _load_rgb(translated.scan.prepared.page_path(tp.index))
        canvas   = _to_rgba(original)

        # Load masks for this page only
        page_masks = MaskStore.load_page(cp, tp.index)

        erase_masks = [
            m
            for tb in tp.bubbles
            if tb.kind != "skip"
            for bm in [page_masks.get(tb.idx)]
            if bm is not None
            for m in bm.erase_masks
        ]
        if erase_masks and runtime.eraser is not None:
            runtime.eraser.erase(canvas, erase_masks)

        clean = canvas[:, :, :3]

        active   = [tb for tb in tp.bubbles if tb.kind != "skip" and tb.translated_text.strip()]
        pg_geom  = geometry.get(tp.index)

        # Build polygon list from scan.npz geometry (not from domain Box)
        geom_by_idx = {bg.bubble_idx: bg for bg in pg_geom.bubbles} if pg_geom else {}
        polygons = [geom_by_idx[tb.idx].polygon for tb in active if tb.idx in geom_by_idx]
        texts    = [tb.translated_text for tb in active if tb.idx in geom_by_idx]
        active   = [tb for tb in active if tb.idx in geom_by_idx]

        result = typoon_render.typoon_render.render(
            original, clean, polygons, texts, original.shape[1]
        )

        active_info = dict(zip((tb.idx for tb in active), result.bubbles))
        rendered_bubbles = tuple(
            RenderedBubble(
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

        rendered_pages.append(RenderedPage(
            source=tp, bubbles=rendered_bubbles, image_path=image_path,
        ))

    return RenderedChapter(source=translated, pages=tuple(rendered_pages))


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
