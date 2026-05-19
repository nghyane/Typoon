"""Page-level inpaint drivers.

  FullPageInpainter  — single backend call on the full page.
    For cv2 TeLeA (uniform bg).

  TiledInpainter     — per-blob crop, backend call, paste back.
    For ML backends (e.g. remote AOT container) where:
      - the model has limited input size (AOT-GAN: bucket ≤ 384 px)
      - small blobs should each get focused context, not a full-page call
      - cost is per-forward, so per-blob is N-forward but each is cheap

Both mutate `canvas` in-place and return nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .backends import InpaintBackend


__all__ = ["PageInpainter", "FullPageInpainter", "TiledInpainter"]


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


class TiledInpainter:
    """Per-blob crop, backend call, paste back.

    For each connected component in page_mask:
      1. Crop bbox + context_px padding (snapped to `snap` px for ONNX).
      2. backend.inpaint(crop_rgb, crop_mask)
      3. Paste only pixels where crop_mask >= 127 back into canvas.

    No fixed tile size — small blobs get small tiles (cheap forward),
    large blobs get larger tiles. Caller's responsibility to ensure the
    backend handles arbitrary sizes (AOT-GAN container at any pad_mod=8).
    """

    _MIN_BLOB_AREA = 5

    def __init__(
        self,
        backend: InpaintBackend,
        *,
        context_px: int = 16,
        snap: int = 8,
    ) -> None:
        self._backend    = backend
        self._context_px = context_px
        self._snap       = snap

    def inpaint_page(self, canvas: np.ndarray, page_mask: np.ndarray) -> None:
        H, W = canvas.shape[:2]
        rgb  = canvas[:, :, :3]

        n, _, stats, _ = cv2.connectedComponentsWithStats(page_mask, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self._MIN_BLOB_AREA:
                continue
            bx, by = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            # Pad + snap to multiple of `snap`
            pad = self._context_px
            x1, y1 = max(0, bx - pad), max(0, by - pad)
            x2, y2 = min(W, bx + bw + pad), min(H, by + bh + pad)
            cw, ch = _snap_up(x2 - x1, self._snap), _snap_up(y2 - y1, self._snap)
            x2 = min(W, x1 + cw); x1 = max(0, x2 - cw)
            y2 = min(H, y1 + ch); y1 = max(0, y2 - ch)

            tile_rgb  = rgb[y1:y2, x1:x2].copy()
            tile_mask = page_mask[y1:y2, x1:x2]
            if not (tile_mask >= 127).any():
                continue

            result = self._backend.inpaint(tile_rgb, tile_mask)
            where  = tile_mask >= 127
            rgb[y1:y2, x1:x2][where] = result[where]
            if canvas.shape[2] == 4:
                canvas[y1:y2, x1:x2, 3][where] = 255


def _snap_up(v: int, snap: int) -> int:
    return ((v + snap - 1) // snap) * snap
