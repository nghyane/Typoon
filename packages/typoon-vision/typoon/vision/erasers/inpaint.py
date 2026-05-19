"""Page-level inpaint drivers.

  FullPageInpainter  — single backend call on the full page.
    For cv2 TeLeA (uniform bg), remote HTTP backends.

Both mutate `canvas` in-place and return nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .backends import InpaintBackend


__all__ = ["PageInpainter", "FullPageInpainter"]


@runtime_checkable
class PageInpainter(Protocol):
    """Fill the masked region of a full page canvas, in place.

    `canvas` is RGBA uint8 (H, W, 4).
    `page_mask` is uint8 (H, W): 255 = inpaint, 0 = keep.
    Pixels outside the mask must not be changed.
    """

    def inpaint_page(
        self,
        canvas: np.ndarray,
        page_mask: np.ndarray,
    ) -> None: ...


class FullPageInpainter:
    """Passes the full RGB page to the backend, pastes result back."""

    def __init__(self, backend: InpaintBackend) -> None:
        self._backend = backend

    def inpaint_page(self, canvas: np.ndarray, page_mask: np.ndarray) -> None:
        rgb    = canvas[:, :, :3]
        result = self._backend.inpaint(rgb.copy(), page_mask)
        where  = page_mask == 255
        rgb[where] = result[where]
        if canvas.shape[2] == 4:
            canvas[:, :, 3][where] = 255
