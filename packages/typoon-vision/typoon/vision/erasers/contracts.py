"""Inpaint backend contracts.

An InpaintBackend takes a cropped image region and a binary mask,
returns the inpainted region. All coordinate math happens in the
caller (HybridEraser); backends only see pixels.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


__all__ = ["InpaintBackend"]


@runtime_checkable
class InpaintBackend(Protocol):
    """Inpaint a masked region of an image.

    Inputs:
      image_rgb  — uint8 (H, W, 3) RGB crop
      mask       — uint8 (H, W)  255=inpaint, 0=keep

    Returns:
      uint8 (H, W, 3) RGB — inpainted result, same size as input.
      Pixels outside the mask should be preserved unchanged.

    Routing hint:
      tile_size — preferred native input side, in page pixels.
        `None`     → backend handles arbitrary page-sized input itself
                     (e.g. cv2.inpaint, remote services). The router
                     calls `inpaint(full_page, full_page_mask)` once.
        `int`      → router crops per-blob tiles of this exact size
                     before calling. Lets local ONNX models keep
                     pixel-perfect native resolution instead of being
                     resized by the page-level path.
    """

    name: str
    tile_size: int | None

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray: ...
