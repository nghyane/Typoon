"""HybridEraser — routes each mask cluster to the right inpaint backend.

Routing logic:
  uniform background (luminance spread < threshold)
    → TeLeA  (cv2, zero-model, ~90ms full page)
  complex background (screentone, art, halftone)
    → complex_backend (AOT-GAN / CF SD1.5 / FLUX2 / …)

Both paths share the same cluster → crop → inpaint → blend pipeline.
Swap complex_backend in config without touching any other code.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

from ..contracts import TextMask
from .backends import AOTGANBackend, InpaintBackend, TeLeABackend
from .routing import build_page_mask, classify_masks, inpaint_region

__all__ = ["HybridEraser"]

logger = logging.getLogger(__name__)


class HybridEraser:
    """Async eraser that dispatches to TeLeA or complex backend per mask.

    uniform_backend  — for low-spread backgrounds (default: TeLeA)
    complex_backend  — for screentone/art backgrounds (default: AOT-GAN)
    spread_threshold — luminance p90-p10 below this → uniform
    """

    name = "hybrid"

    def __init__(
        self,
        *,
        uniform_backend: InpaintBackend | None = None,
        complex_backend: InpaintBackend | None = None,
        spread_threshold: int = 30,
    ) -> None:
        self._uniform   = uniform_backend  or TeLeABackend()
        self._complex   = complex_backend  # None → falls back to uniform
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
        uniform_masks, complex_masks = classify_masks(
            canvas, masks, self._threshold
        )
        if uniform_masks:
            self._erase_batch(canvas, uniform_masks, self._uniform)
        if complex_masks:
            backend = self._complex or self._uniform
            self._erase_batch(canvas, complex_masks, backend)

    def _erase_batch(
        self,
        canvas: np.ndarray,
        masks: list[TextMask],
        backend: InpaintBackend,
    ) -> None:
        """Build combined page mask for this batch and call backend."""
        ch, cw = canvas.shape[:2]
        page_mask = build_page_mask(masks, cw, ch)
        if not (page_mask > 0).any():
            return
        try:
            inpaint_region(canvas, page_mask, backend)
        except Exception:
            logger.warning(
                "HybridEraser: %s failed, falling back to TeLeA",
                backend.name, exc_info=True,
            )
            inpaint_region(canvas, page_mask, TeLeABackend())
