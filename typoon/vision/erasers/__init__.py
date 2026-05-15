"""Eraser backends — remove text from a page canvas.

AOT-GAN inpainting (default) with median fill fallback. Both are async-
native; sync model inference runs in asyncio.to_thread so the event loop
stays responsive.

Logic carried over from legacy vision/erase.py with the same algorithm
(cluster masks → AOT-GAN per cluster → blend); only the I/O surface is
async + Protocol-conformant.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

from ..contracts import TextMask


__all__ = ["AOTGANEraser", "MedianEraser"]

logger = logging.getLogger(__name__)


_MIN_CONTEXT_PAD = 64
_INPAINT_MAX_DIM = 512
_CONTEXT_RATIO = 0.5
_CLUSTER_MAX_DIM = _INPAINT_MAX_DIM * 2


# ─── AOT-GAN ──────────────────────────────────────────────────────────────


class AOTGANEraser:
    """AOT-GAN inpainting with median fallback per cluster.

    Lazy model load on first erase(). Inference releases GIL → multiple
    pages can erase concurrently up to runtime.spec.erase_concurrency.
    """

    name = "aot_gan"

    def __init__(self, *, models_dir: Path) -> None:
        self._models_dir = Path(models_dir)
        self._inpainter = None

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
        ch, cw = canvas.shape[:2]
        inpainter = self._get_inpainter()
        if inpainter is None:
            for m in masks:
                _erase_with_median(canvas, m)
            return
        for cluster in _cluster_masks(masks):
            try:
                _erase_cluster_with_aot(canvas, cluster, cw, ch, inpainter)
            except Exception:
                logger.warning("AOT inpaint failed, falling back to median", exc_info=True)
                for m in cluster:
                    _erase_with_median(canvas, m)

    def _get_inpainter(self):
        if self._inpainter is not None:
            return self._inpainter
        try:
            from .._backends.aot import AOTInpainter
        except ImportError:
            return None
        try:
            self._inpainter = AOTInpainter(str(self._models_dir))
        except Exception:
            logger.warning("AOT model unavailable; falling back to median", exc_info=True)
            return None
        return self._inpainter


# ─── Median (fallback / no-model) ─────────────────────────────────────────


class MedianEraser:
    """Pure median-fill eraser. No model needed; fast but visible."""

    name = "median_only"

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
        for m in masks:
            _erase_with_median(canvas, m)


# ─── AOT cluster inpaint ──────────────────────────────────────────────────


def _erase_cluster_with_aot(
    canvas: np.ndarray,
    cluster: list[TextMask],
    cw: int, ch: int,
    inpainter,
) -> None:
    crop_x1, crop_y1, crop_x2, crop_y2 = _cluster_crop(cluster, cw, ch)
    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

    combined = np.zeros((crop_h, crop_w), dtype=np.uint8)
    for m in cluster:
        _blit_mask(combined, m, crop_x1, crop_y1)

    crop_rgb = canvas[crop_y1:crop_y2, crop_x1:crop_x2, :3].copy()

    long_side = max(crop_w, crop_h)
    scale = None
    if long_side > _INPAINT_MAX_DIM:
        scale = _INPAINT_MAX_DIM / long_side
        nw, nh = round(crop_w * scale), round(crop_h * scale)
        inf_rgb = cv2.resize(crop_rgb, (nw, nh))
        inf_mask = cv2.resize(combined, (nw, nh), interpolation=cv2.INTER_NEAREST)
    else:
        inf_rgb, inf_mask = crop_rgb, combined

    result = inpainter.inpaint(inf_rgb, inf_mask)

    if scale is not None:
        result = cv2.resize(result, (crop_w, crop_h))

    _blend_inpainted_cluster(
        canvas, result, combined, crop_x1, crop_y1, crop_w, crop_h,
    )


# ─── Median-fill primitive ────────────────────────────────────────────────


def _erase_with_median(canvas: np.ndarray, mask: TextMask) -> None:
    color, _ = _sample_background(canvas, mask)
    mh = mask.image.shape[0]
    pad = mh // 20
    apply = _dilate_into_larger(mask, pad) if pad > 0 else mask
    _apply_color(canvas, apply, color)


def _sample_background(img: np.ndarray, mask: TextMask) -> tuple[np.ndarray, int]:
    ih, iw = img.shape[:2]
    mh, mw = mask.image.shape[:2]

    region = img[mask.y:mask.y + mh:2, mask.x:mask.x + mw:2]
    mask_sub = mask.image[::2, ::2]
    bg = region[mask_sub == 0]
    if len(bg) == 0:
        return np.array([255, 255, 255, 255], dtype=np.uint8), 0
    median = np.median(bg, axis=0).astype(np.uint8)
    if median.shape[0] == 3:
        median = np.append(median, 255)

    pad = _context_pad(mw, mh)
    ey1, ey2 = max(mask.y - pad, 0), min(mask.y + mh + pad, ih)
    ex1, ex2 = max(mask.x - pad, 0), min(mask.x + mw + pad, iw)
    expanded = img[ey1:ey2:2, ex1:ex2:2]
    lums = (
        expanded[..., 0].astype(np.uint32) * 299
        + expanded[..., 1].astype(np.uint32) * 587
        + expanded[..., 2].astype(np.uint32) * 114
    ) // 1000
    flat = np.sort(lums.ravel().astype(np.uint8))
    if len(flat) == 0:
        return median, 0
    spread = int(flat[len(flat) * 9 // 10]) - int(flat[len(flat) // 10])
    return median, max(spread, 0)


def _apply_color(canvas: np.ndarray, mask: TextMask, color: np.ndarray) -> None:
    ch, cw = canvas.shape[:2]
    mh, mw = mask.image.shape[:2]
    y1, y2 = max(mask.y, 0), min(mask.y + mh, ch)
    x1, x2 = max(mask.x, 0), min(mask.x + mw, cw)
    if y1 >= y2 or x1 >= x2:
        return
    my1, my2 = y1 - mask.y, y2 - mask.y
    mx1, mx2 = x1 - mask.x, x2 - mask.x
    where = mask.image[my1:my2, mx1:mx2] == 255
    canvas[y1:y2, x1:x2][where] = color


# ─── Cluster + crop helpers ───────────────────────────────────────────────


def _context_pad(mask_w: int, mask_h: int) -> int:
    return max(int(max(mask_w, mask_h) * _CONTEXT_RATIO), _MIN_CONTEXT_PAD)


def _mask_bbox(m: TextMask) -> tuple[int, int, int, int]:
    mh, mw = m.image.shape[:2]
    return m.x, m.y, m.x + mw, m.y + mh


def _union_bbox(bboxes):
    return (
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    )


def _bbox_gap(a, b) -> int:
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return dx + dy


def _cluster_masks(masks: list[TextMask]) -> list[list[TextMask]]:
    if not masks:
        return []
    clusters: list[list[TextMask]] = [[m] for m in masks]
    bboxes = [_mask_bbox(m) for m in masks]
    while len(clusters) > 1:
        best_gap, best_i, best_j = None, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                combined = _union_bbox([bboxes[i], bboxes[j]])
                if max(combined[2]-combined[0], combined[3]-combined[1]) > _CLUSTER_MAX_DIM:
                    continue
                gap = _bbox_gap(bboxes[i], bboxes[j])
                if best_gap is None or gap < best_gap:
                    best_gap, best_i, best_j = gap, i, j
        if best_i == -1:
            break
        clusters[best_i] = clusters[best_i] + clusters.pop(best_j)
        bboxes[best_i] = _union_bbox([bboxes[best_i], bboxes.pop(best_j)])
    return clusters


def _cluster_crop(masks, cw, ch):
    mx1 = min(m.x for m in masks)
    my1 = min(m.y for m in masks)
    mx2 = max(m.x + m.image.shape[1] for m in masks)
    my2 = max(m.y + m.image.shape[0] for m in masks)
    pad = _context_pad(mx2 - mx1, my2 - my1)
    return (
        max(mx1 - pad, 0), max(my1 - pad, 0),
        min(mx2 + pad, cw), min(my2 + pad, ch),
    )


def _blit_mask(target, mask: TextMask, crop_x: int, crop_y: int) -> None:
    th, tw = target.shape[:2]
    mh, mw = mask.image.shape[:2]
    ox, oy = mask.x - crop_x, mask.y - crop_y
    sx, sy = max(-ox, 0), max(-oy, 0)
    ex, ey = min(mw, tw - ox), min(mh, th - oy)
    if sx >= ex or sy >= ey:
        return
    target[oy + sy:oy + ey, ox + sx:ox + ex] |= mask.image[sy:ey, sx:ex]


def _blend_inpainted_cluster(
    canvas, inpainted, combined_mask,
    crop_x, crop_y, crop_w, crop_h,
) -> None:
    ch, cw = canvas.shape[:2]
    cx1 = max(crop_x, 0); cy1 = max(crop_y, 0)
    cx2 = min(crop_x + crop_w, cw); cy2 = min(crop_y + crop_h, ch)
    if cx1 >= cx2 or cy1 >= cy2:
        return
    lx1, ly1 = cx1 - crop_x, cy1 - crop_y
    lx2, ly2 = cx2 - crop_x, cy2 - crop_y
    where = combined_mask[ly1:ly2, lx1:lx2] == 255
    region = canvas[cy1:cy2, cx1:cx2]
    inp = inpainted[ly1:ly2, lx1:lx2]
    region[where, :3] = inp[where]
    region[where, 3] = 255


def _dilate_into_larger(mask: TextMask, pad: int) -> TextMask:
    mh, mw = mask.image.shape[:2]
    expanded = np.zeros((mh + pad * 2, mw + pad * 2), dtype=np.uint8)
    expanded[pad:pad + mh, pad:pad + mw] = mask.image
    ksize = pad * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(expanded, kernel, iterations=1)
    new_x = mask.x - pad
    new_y = mask.y - pad
    crop_x = max(-new_x, 0)
    crop_y = max(-new_y, 0)
    return TextMask(
        x=max(new_x, 0),
        y=max(new_y, 0),
        image=dilated[crop_y:, crop_x:],
    )
