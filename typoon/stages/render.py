"""Render stage — TranslatedChapter + MaskStore → RenderedChapter.

Pipeline per page:
  1. Load prepared page image (RGB)
  2. Erase source text via MaskStore + VisionRuntime.eraser → clean RGB
  3. Render translated text via typoon_render.render()
  4. Write PNG to render_dir/
  5. Return RenderedChapter with per-bubble font metrics
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain.render import Bubble as RenderedBubble, Chapter as RenderedChapter, Page as RenderedPage
from typoon.domain.translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter
from typoon.runs.artifacts import ArtifactSink


def render_chapter(
    translated: TranslatedChapter,
    masks: MaskStore,
    runtime: VisionRuntime,
    *,
    render_dir: Path | None = None,
    artifacts: ArtifactSink | None = None,
) -> RenderedChapter:
    """Erase source text, render translations, write PNGs to render_dir."""
    if render_dir is not None:
        render_dir = Path(render_dir)
        render_dir.mkdir(parents=True, exist_ok=True)

    import typoon_render

    rendered_pages = []

    for tp in translated.pages:
        original = _load_rgb(translated.scan.prepared.page_path(tp.index))
        canvas   = _to_rgba(original)

        # Collect erase masks for accepted bubbles
        erase_masks = [
            m
            for tb in tp.bubbles
            if tb.kind != "skip"
            for bm in [masks.get(tb.page_index, tb.idx)]
            if bm is not None
            for m in bm.erase_masks
        ]
        if erase_masks and runtime.eraser is not None:
            runtime.eraser.erase(canvas, erase_masks)

        clean = canvas[:, :, :3]

        # Render text — only non-skip bubbles with actual translation
        active = [tb for tb in tp.bubbles if tb.kind != "skip" and tb.translated_text.strip()]
        polygons = [tb.source.box.polygon for tb in active]
        texts    = [tb.translated_text for tb in active]

        result = typoon_render.typoon_render.render(
            original, clean, polygons, texts, original.shape[1]
        )

        # Map render results back to all bubbles
        active_info = dict(zip((tb.idx for tb in active), result.bubbles))
        rendered_bubbles = tuple(
            RenderedBubble(
                source=tb,
                font_size=active_info[tb.idx].font_size_px if tb.idx in active_info else 0,
                overflow=active_info[tb.idx].overflow if tb.idx in active_info else False,
            )
            for tb in tp.bubbles
        )

        image_path = None
        if render_dir is not None:
            image_path = render_dir / f"page_{tp.index:04d}.png"
            _write_rgb(image_path, result.image)

        if artifacts is not None:
            artifacts.write_image("06_render", f"page_{tp.index:04d}_rendered.png", result.image)

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
