"""TeLeA backend — cv2.inpaint (INPAINT_TELEA).

Fast Marching Method: propagates colour from the mask boundary inward.
Zero model weight, ~90ms for a full manga page on CPU.

Best for: uniform backgrounds (white speech bubbles, caption boxes).
Not suitable for: screentone, halftone, complex art texture.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import InpaintBackend

__all__ = ["TeLeABackend"]


class TeLeABackend:
    """cv2 TELEA inpainting — no model, pure algorithm."""

    name = "telea"
    tile_size: int | None = None  # cv2 handles full page in ~90ms; no tiling

    def __init__(self, radius: int = 10) -> None:
        self._radius = radius

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        bgr    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = cv2.inpaint(bgr, mask, self._radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
