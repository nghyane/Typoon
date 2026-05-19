"""Page-level inpaint drivers.

Two concrete implementations of PageInpainter:

  FullPageInpainter — single backend call on the full page.
    Suited to algorithms that scale naturally to page size (cv2 TeLeA,
    remote HTTP services).

  TiledInpainter — per-blob native-resolution crops, then paste back.
    Suited to local ML backends bounded to a fixed input dimension
    (e.g. AOT-GAN bucket ≤ 384 px). Each blob gets a square crop of
    `tile_size` pixels centred on it so the model sees real page
    resolution instead of a downscaled full page.

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
    """Passes the full RGB page to the backend, pastes result back.

    Used for TeLeA and remote backends that handle arbitrary page sizes.
    """

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
    """Per-blob tile crops at native page resolution, then paste back.

    Algorithm (mirrors manga-cleaner's tile-per-blob approach):
      1. connectedComponentsWithStats on page_mask → blobs
      2. Sort blobs by area descending (large blobs first).
      3. For each blob not yet fully covered:
           a. Compute tile_size × tile_size crop centred on the blob,
              clamped to page bounds.
           b. Call backend.inpaint(tile_rgb, tile_mask).
           c. Paste only where tile_mask >= 127 back into canvas.
           d. Mark tile region as covered.
      4. Skip blobs whose entire bbox is already covered.

    Because each forward only sees a small native-resolution crop, the
    model is never asked to downscale a full page. For AOT-GAN this
    keeps the input inside the 384 px bucket and avoids the
    downscale → blur → upscale artefact visible on screentone.
    """

    _MIN_BLOB_AREA = 5  # px² — noise gate

    def __init__(self, backend: InpaintBackend, *, tile_size: int) -> None:
        self._backend   = backend
        self._tile_size = tile_size

    def inpaint_page(self, canvas: np.ndarray, page_mask: np.ndarray) -> None:
        H, W = canvas.shape[:2]
        rgb  = canvas[:, :, :3]

        n, _, stats, _ = cv2.connectedComponentsWithStats(page_mask, connectivity=8)
        blobs = sorted(
            (stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > self._MIN_BLOB_AREA),
            key=lambda s: -s[cv2.CC_STAT_AREA],
        )

        covered = np.zeros_like(page_mask)

        for s in blobs:
            bx, by, bw, bh = s[cv2.CC_STAT_LEFT], s[cv2.CC_STAT_TOP], s[cv2.CC_STAT_WIDTH], s[cv2.CC_STAT_HEIGHT]
            if (covered[by:by + bh, bx:bx + bw] == 255).all():
                continue

            x1, y1, x2, y2 = _center_tile(bx, by, bw, bh, self._tile_size, W, H)
            tile_mask = page_mask[y1:y2, x1:x2]
            if not tile_mask.any():
                continue

            tile_rgb = rgb[y1:y2, x1:x2].copy()
            result   = self._backend.inpaint(tile_rgb, tile_mask)

            where = tile_mask >= 127
            rgb[y1:y2, x1:x2][where] = result[where]
            if canvas.shape[2] == 4:
                canvas[y1:y2, x1:x2, 3][where] = 255
            covered[y1:y2, x1:x2] = 255


def _center_tile(
    bx: int, by: int, bw: int, bh: int,
    T: int, W: int, H: int,
) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) for a T×T square centred on the blob."""
    cx, cy = bx + bw // 2, by + bh // 2
    x1 = max(0, cx - T // 2);  y1 = max(0, cy - T // 2)
    x2 = min(W, x1 + T);       y2 = min(H, y1 + T)
    x1 = max(0, x2 - T);       y1 = max(0, y2 - T)
    return x1, y1, x2, y2
