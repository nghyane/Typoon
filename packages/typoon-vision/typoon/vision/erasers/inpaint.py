"""Page-level inpaint drivers.

Two concrete implementations of PageInpainter:

  FullPageInpainter — single backend call on the full page.
    Suited to algorithms that scale naturally to page size (cv2 TeLeA,
    remote HTTP services).

  TiledInpainter — per-blob adaptive-resolution crops, then paste back.
    Suited to local ML backends bounded to a fixed input dimension
    (e.g. AOT-GAN bucket ≤ 384 px). Each blob gets a crop of
    bbox + context_px padding (snapped to `snap` px) so the model sees
    real page resolution instead of a downscaled full page. Adaptive
    sizing keeps small bubbles in cheap buckets (128/192/256) rather
    than always paying the 384 forward cost.

Both mutate `canvas` in-place and return nothing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from .backends import InpaintBackend


__all__ = ["PageInpainter", "FullPageInpainter", "TiledInpainter", "AreaGatedInpainter"]


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
    """Per-blob adaptive-tile crops at native page resolution, then paste back.

    Algorithm:
      1. connectedComponentsWithStats on page_mask → blobs
      2. Sort blobs by area descending (large blobs first).
      3. For each blob not yet fully covered:
           a. Compute an adaptive crop: blob bbox + context_px padding on
              each side, rounded up to a multiple of snap px (ONNX mod
              requirement). No fixed upper cap — large bubbles get a larger
              tile, keeping AOT-GAN in its low-cost buckets for small bubbles.
           b. Call backend.inpaint(tile_rgb, tile_mask).
           c. Paste only where tile_mask >= 127 back into canvas.
           d. Mark tile region as covered.
      4. Skip blobs whose entire bbox is already covered.

    Adaptive sizing vs fixed 384:
      A typical manga bubble is 80-150 px. With context_px=64 that yields
      a 200-280 px tile → AOT-GAN bucket 256 (~3s, ~4GB) instead of 384
      (~21s, ~7GB). Quality is identical — the model still sees the full
      bubble plus enough background context for screentone reconstruction.
    """

    _MIN_BLOB_AREA = 5  # px² — noise gate

    def __init__(
        self,
        backend: InpaintBackend,
        *,
        context_px: int = 64,
        snap: int = 8,
    ) -> None:
        self._backend    = backend
        self._context_px = context_px
        self._snap       = snap

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
            bx = s[cv2.CC_STAT_LEFT];  by = s[cv2.CC_STAT_TOP]
            bw = s[cv2.CC_STAT_WIDTH]; bh = s[cv2.CC_STAT_HEIGHT]
            if (covered[by:by + bh, bx:bx + bw] == 255).all():
                continue

            x1, y1, x2, y2 = _adaptive_tile(
                bx, by, bw, bh, W, H,
                context_px=self._context_px,
                snap=self._snap,
            )
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


# ─── tile geometry helpers ─────────────────────────────────────────────────


def _adaptive_tile(
    bx: int, by: int, bw: int, bh: int,
    W: int, H: int,
    *,
    context_px: int,
    snap: int,
) -> tuple[int, int, int, int]:
    """Tight crop: blob bbox + context padding, snapped to `snap`.

    Padding is max(context_px, 25% of blob dimension) so tiny blobs still
    get meaningful context and large blobs don't over-pad. Width and height
    are rounded up to the next multiple of snap (ONNX mod requirement),
    then centred on the blob and clamped to page bounds.
    """
    pad = max(context_px, bw // 4, bh // 4)
    tw  = _snap_up(bw + pad * 2, snap)
    th  = _snap_up(bh + pad * 2, snap)
    cx  = bx + bw // 2;  cy = by + bh // 2
    x1  = max(0, cx - tw // 2);  y1 = max(0, cy - th // 2)
    x2  = min(W, x1 + tw);       y2 = min(H, y1 + th)
    x1  = max(0, x2 - tw);       y1 = max(0, y2 - th)
    return x1, y1, x2, y2


def _snap_up(v: int, snap: int) -> int:
    """Round v up to the next multiple of snap."""
    return ((v + snap - 1) // snap) * snap


def _center_tile(
    bx: int, by: int, bw: int, bh: int,
    T: int, W: int, H: int,
) -> tuple[int, int, int, int]:
    """Fixed T×T square centred on blob, clamped to page. Used in tests."""
    cx, cy = bx + bw // 2, by + bh // 2
    x1 = max(0, cx - T // 2);  y1 = max(0, cy - T // 2)
    x2 = min(W, x1 + T);       y2 = min(H, y1 + T)
    x1 = max(0, x2 - T);       y1 = max(0, y2 - T)
    return x1, y1, x2, y2


class AreaGatedInpainter:
    """Route each blob to cheap or ML inpainter based on mask area.

    Blobs with mask_area < area_threshold px² go to `small_inpainter`
    (e.g. TeLeA — fast, zero model cost). Larger blobs go to
    `large_inpainter` (e.g. TiledInpainter + AOT-GAN).

    Rationale: SFX text and small floating kanji rarely exceed 1000 px²
    of masked area. Quality difference between TeLeA and ML is invisible
    at that scale. Dialogue bubbles are typically 3000-15000 px² where ML
    quality is worth the cost.

    Runs per-blob by iterating connected components, same as TiledInpainter.
    """

    _MIN_BLOB_AREA = 5

    def __init__(
        self,
        *,
        small_inpainter: PageInpainter,
        large_inpainter: PageInpainter,
        area_threshold: int = 1000,
    ) -> None:
        self._small     = small_inpainter
        self._large     = large_inpainter
        self._threshold = area_threshold

    def inpaint_page(self, canvas: np.ndarray, page_mask: np.ndarray) -> None:
        n, _, stats, _ = cv2.connectedComponentsWithStats(page_mask, connectivity=8)
        blobs = [stats[i] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > self._MIN_BLOB_AREA]

        small_mask = np.zeros_like(page_mask)
        large_mask = np.zeros_like(page_mask)

        for s in blobs:
            bx = s[cv2.CC_STAT_LEFT]; by = s[cv2.CC_STAT_TOP]
            bw = s[cv2.CC_STAT_WIDTH]; bh = s[cv2.CC_STAT_HEIGHT]
            area = s[cv2.CC_STAT_AREA]
            target = small_mask if area < self._threshold else large_mask
            target[by:by+bh, bx:bx+bw] |= page_mask[by:by+bh, bx:bx+bw]

        if (small_mask > 0).any():
            self._small.inpaint_page(canvas, small_mask)
        if (large_mask > 0).any():
            self._large.inpaint_page(canvas, large_mask)
