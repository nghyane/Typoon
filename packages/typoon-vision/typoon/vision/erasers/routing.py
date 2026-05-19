"""Routing helpers for HybridEraser.

classify_masks  — split masks into (uniform, complex) lists
build_page_mask — stamp masks onto a full-page binary mask
inpaint_region  — crop → inpaint → paste only masked pixels back
"""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import TextMask
from .backends import InpaintBackend


__all__ = ["classify_masks", "build_page_mask", "inpaint_region"]


# Alias for probe scripts / backward compat
def _is_uniform_background(canvas, cluster, spread_threshold=30):
    """Single-cluster wrapper for classify_masks."""
    u, _ = classify_masks(canvas, list(cluster), spread_threshold)
    return len(u) == len(list(cluster))


def classify_masks(
    canvas: np.ndarray,
    masks: list[TextMask],
    spread_threshold: int,
) -> tuple[list[TextMask], list[TextMask]]:
    """Split masks into (uniform_bg, complex_bg) lists.

    Uniformity is measured per-mask: sample pixels inside the mask bbox
    but OUTSIDE the mask itself (background pixels), compute luminance
    p90-p10 spread. Low spread → flat colour → uniform.
    """
    uniform: list[TextMask] = []
    complex_: list[TextMask] = []
    for m in masks:
        if _is_uniform(canvas, m, spread_threshold):
            uniform.append(m)
        else:
            complex_.append(m)
    return uniform, complex_


def _is_uniform(
    canvas: np.ndarray,
    mask: TextMask,
    threshold: int,
) -> bool:
    ch, cw = canvas.shape[:2]
    mx, my = mask.x, mask.y
    mh, mw = mask.image.shape[:2]
    tx1, ty1 = max(0, mx), max(0, my)
    tx2, ty2 = min(cw, mx + mw), min(ch, my + mh)
    if tx2 <= tx1 or ty2 <= ty1:
        return True
    sx, sy = tx1 - mx, ty1 - my
    crop_canvas = canvas[ty1:ty2, tx1:tx2, :3]
    crop_mask   = mask.image[sy:sy + (ty2 - ty1), sx:sx + (tx2 - tx1)]
    bg = crop_canvas[crop_mask == 0]
    if len(bg) < 10:
        return True
    lum = (
        bg[:, 0].astype(np.int32) * 299
        + bg[:, 1].astype(np.int32) * 587
        + bg[:, 2].astype(np.int32) * 114
    ) // 1000
    flat   = np.sort(lum.ravel().astype(np.uint8))
    spread = int(flat[len(flat) * 9 // 10]) - int(flat[len(flat) // 10])
    return spread < threshold


def build_page_mask(
    masks: list[TextMask],
    page_w: int,
    page_h: int,
) -> np.ndarray:
    """OR all masks onto a (H, W) uint8 page-level mask."""
    pm = np.zeros((page_h, page_w), dtype=np.uint8)
    for m in masks:
        mx, my = m.x, m.y
        mh, mw = m.image.shape[:2]
        tx1, ty1 = max(0, mx), max(0, my)
        tx2, ty2 = min(page_w, mx + mw), min(page_h, my + mh)
        if tx2 <= tx1 or ty2 <= ty1:
            continue
        sx, sy = tx1 - mx, ty1 - my
        pm[ty1:ty2, tx1:tx2] |= m.image[sy:sy + (ty2 - ty1), sx:sx + (tx2 - tx1)]
    return pm


def inpaint_region(
    canvas: np.ndarray,
    page_mask: np.ndarray,
    backend: InpaintBackend,
) -> None:
    """Call backend on full-page mask, paste only masked pixels back.

    canvas is RGBA; backend works on RGB. Alpha channel preserved.
    Only pixels where page_mask==255 are overwritten.
    """
    rgb    = canvas[:, :, :3].copy()
    result = backend.inpaint(rgb, page_mask)
    where  = page_mask == 255
    canvas[:, :, :3][where] = result[where]
    if canvas.shape[2] == 4:
        canvas[:, :, 3][where] = 255
