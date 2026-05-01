"""Render stage — TranslatedChapter + MaskStore → RenderedChapter.

Pipeline per page:
  1. Load prepared page image (RGB)
  2. Convert to RGBA canvas
  3. Erase source text via MaskStore + VisionRuntime.eraser
  4. Render translated text with TextRenderer
  5. Write PNG to out_dir/render/
  6. Return RenderedChapter
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from typoon.adapters.mask_store import MaskStore
from typoon.adapters.text_renderer import render_text
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain.render import Bubble as RenderedBubble, Chapter as RenderedChapter, Page as RenderedPage
from typoon.domain.translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter
from typoon.runs.artifacts import ArtifactSink


def render_chapter(
    translated: TranslatedChapter,
    masks: MaskStore,
    runtime: VisionRuntime,
    *,
    out_dir: Path | None = None,
    artifacts: ArtifactSink | None = None,
) -> RenderedChapter:
    """Erase source text, render translations, write PNGs."""
    if out_dir is not None:
        out_dir = Path(out_dir) / "render"
        out_dir.mkdir(parents=True, exist_ok=True)

    rendered_pages = []

    for tp in translated.pages:
        image = _load_rgb(translated.scan.prepared.page_path(tp.index))
        canvas = _to_rgba(image)

        # Erase
        erase_masks = []
        for tb in tp.bubbles:
            if tb.kind == "skip":
                continue
            bm = masks.get(tb.page_index, tb.idx)
            if bm is not None:
                erase_masks.extend(bm.erase_masks)
        if erase_masks and runtime.eraser is not None:
            runtime.eraser.erase(canvas, list(erase_masks))

        # Render text
        rendered_bubbles = []
        for tb in tp.bubbles:
            font_size, overflow = 0, False
            if tb.kind != "skip" and tb.translated_text.strip():
                font_size, overflow = render_text(
                    canvas, tb.translated_text, tb.source.box.fit
                )
            rendered_bubbles.append(RenderedBubble(source=tb, font_size=font_size, overflow=overflow))

        result = canvas[:, :, :3]

        # Write PNG
        if out_dir is not None:
            tag = f"page_{tp.index:04d}.png"
            _write_rgb(out_dir / tag, result)

        if artifacts is not None:
            artifacts.write_image("06_render", f"page_{tp.index:04d}_rendered.png", result)

        rendered_pages.append(RenderedPage(
            source=tp, bubbles=tuple(rendered_bubbles), image=result,
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
