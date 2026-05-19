"""TextEraser — production text eraser.

Routes each TextMask to the appropriate PageInpainter based on
background uniformity (luminance p90-p10 spread).

  uniform background (spread < threshold)
    → uniform_inpainter  (default: FullPageInpainter + TeLeABackend)
      Fast, zero model, ~90ms/page; right tool for white speech bubbles.

  complex background (screentone, halftone, art)
    → complex_inpainter  (default: TiledInpainter + AOTGANBackend 384px)
      Per-blob native-resolution crops; model never sees a downscaled
      full page.

Both inpainters receive a combined page-level mask for their batch.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from ..contracts import TextMask
from .inpaint import FullPageInpainter, PageInpainter, TiledInpainter
from .routing import build_page_mask, partition_by_background

__all__ = ["TextEraser"]

logger = logging.getLogger(__name__)


class TextEraser:
    """Async eraser — TeLeA for uniform bg, tiled AOT-GAN for complex bg.

    `uniform_inpainter` and `complex_inpainter` are injected so the
    routing strategy is testable without touching erasure logic and
    swappable (e.g. substitute LaMa for complex) without subclassing.
    """

    name = "text"

    def __init__(
        self,
        *,
        uniform_inpainter: PageInpainter,
        complex_inpainter: PageInpainter,
        spread_threshold: int = 30,
    ) -> None:
        self._uniform   = uniform_inpainter
        self._complex   = complex_inpainter
        self._threshold = spread_threshold

    async def erase(
        self,
        canvas: np.ndarray,
        masks: tuple[TextMask, ...],
    ) -> np.ndarray:
        if not masks:
            return canvas
        await asyncio.to_thread(self._erase_sync, canvas, list(masks))
        return canvas

    def _erase_sync(self, canvas: np.ndarray, masks: list[TextMask]) -> None:
        H, W = canvas.shape[:2]
        uniform, complex_ = partition_by_background(canvas, masks, self._threshold)

        if uniform:
            pm = build_page_mask(uniform, W, H)
            if (pm > 0).any():
                try:
                    self._uniform.inpaint_page(canvas, pm)
                except Exception:
                    logger.warning("uniform inpainter failed", exc_info=True)

        if complex_:
            pm = build_page_mask(complex_, W, H)
            if (pm > 0).any():
                try:
                    self._complex.inpaint_page(canvas, pm)
                except Exception:
                    logger.warning(
                        "complex inpainter failed, falling back to uniform",
                        exc_info=True,
                    )
                    self._uniform.inpaint_page(canvas, pm)
