"""Render stage — TranslatedChapter + MaskStore → RenderedChapter.

Pipeline per page:
  1. Load prepared page image
  2. For each accepted bubble: erase source text with MaskStore masks
  3. For each bubble with translated text: layout + rasterize text
  4. Return RenderedChapter with composited images

Text layout is a stub — returns placeholder until a renderer is wired in.
Erase uses VisionRuntime.eraser (AOT-GAN inpainting or median fallback).
"""

from __future__ import annotations

import cv2
import numpy as np

from typoon.adapters.mask_store import BubbleMasks, MaskStore
from typoon.adapters.vision_runtime import VisionRuntime
from typoon.domain.render import Bubble as RenderedBubble, Chapter as RenderedChapter, Page as RenderedPage
from typoon.domain.translate import TranslatedBubble, TranslatedChapter
from typoon.runs.artifacts import ArtifactSink


def render_chapter(
    translated: TranslatedChapter,
    masks: MaskStore,
    runtime: VisionRuntime,
    *,
    artifacts: ArtifactSink | None = None,
) -> RenderedChapter:
    """Erase source text and render translations onto each page."""
    rendered_pages = []

    for tp in translated.pages:
        image = _load_rgb(translated.scan.prepared.page_path(tp.index))
        canvas = _to_rgba(image)

        # Erase step — apply inpainting for accepted bubbles
        erase_masks = []
        for tb in tp.bubbles:
            if tb.kind == "skip":
                continue
            bm = masks.get(tb.page_index, tb.idx)
            if bm is not None:
                erase_masks.extend(bm.erase_masks)

        if erase_masks and runtime.eraser is not None:
            runtime.eraser.erase(canvas, list(erase_masks))

        # Text render step (stub — real renderer to be wired in)
        rendered_bubbles = tuple(
            _render_bubble(tb, canvas)
            for tb in tp.bubbles
        )

        result_rgb = canvas[:, :, :3]

        if artifacts is not None:
            tag = f"page_{tp.index:04d}"
            artifacts.write_image("06_render", f"{tag}_rendered.png", result_rgb)

        rendered_pages.append(RenderedPage(
            source=tp,
            bubbles=rendered_bubbles,
            image=result_rgb,
        ))

    return RenderedChapter(source=translated, pages=tuple(rendered_pages))


# ── Helpers ──────────────────────────────────────────────────────


def _render_bubble(tb: TranslatedBubble, canvas: np.ndarray) -> RenderedBubble:
    """Stub: placeholder until a real text layout engine is wired in."""
    return RenderedBubble(source=tb, font_size=0, overflow=False)


def _load_rgb(path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read prepared page: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_rgba(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.dstack([image, np.full((h, w), 255, dtype=np.uint8)])
