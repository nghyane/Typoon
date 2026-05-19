"""CTD mask augmenter — replace Lens dilated masks with CTD pixel masks.

Lens gives excellent detection + OCR but its erase masks are
word-bbox-union + dilation — they sometimes clip art or leave residue.
CTD gives pixel-accurate UNet+DBNet masks but lower recall.

This module runs CTD in parallel with Lens, then replaces each Lens
BubbleGroup's masks with the matching CTD pixel mask (IoU ≥ threshold).
Groups with no CTD match keep the original Lens masks unchanged.

Usage:
    augmenter = CTDMaskAugmenter(onnx_path)
    ctd_result = await augmenter.detect(image)
    groups = augmenter.augment(groups, ctd_result)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np

from ..contracts import BubbleGroup, TextMask


__all__ = ["CTDMaskAugmenter"]

logger = logging.getLogger(__name__)

_IOU_MATCH_THRESHOLD = 0.30   # loose — CTD bbox is sometimes slightly smaller
_CLOSE_RADIUS        = 3      # morphological close to fill stroke gaps


class CTDMaskAugmenter:
    """Run CTD alongside Lens to provide pixel-accurate masks.

    Lazy-loads ONNX backend on first augment() call.
    """

    def __init__(self, onnx_path: Path | str) -> None:
        self._onnx_path = Path(onnx_path)
        self._backend   = None

    async def detect(self, image: np.ndarray):
        """Run CTD inference and return CTDResult."""
        backend = self._get_backend()
        return await asyncio.to_thread(backend.detect, image)

    def augment(
        self,
        groups: tuple[BubbleGroup, ...],
        ctd_result,
        image_shape: tuple[int, int] | None = None,
    ) -> tuple[BubbleGroup, ...]:
        """Replace Lens masks with CTD pixel masks where IoU ≥ threshold.

        Returns a new tuple of BubbleGroups. Groups with no CTD match
        are returned unchanged. Also stores the page bubble_mask on
        ctd_result for downstream use.
        """
        if not ctd_result.text_regions:
            return groups

        augmented: list[BubbleGroup] = []
        used_indices: set[int] = set()   # each CTD region claims at most one group
        for g in groups:
            best_region, best_idx = _best_match(g.bbox, ctd_result.text_regions, used_indices)
            if best_region is None:
                augmented.append(g)
                continue

            used_indices.add(best_idx)

            # Build pixel masks from CTD text_mask
            raw = best_region.text_mask            # uint8 (H, W) in bbox coords
            k   = _CLOSE_RADIUS * 2 + 1
            kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            refined = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)

            text_mask  = TextMask(x=best_region.mask_x, y=best_region.mask_y, image=refined)
            erase_mask = text_mask   # CTD mask is already tight — no extra dilation

            augmented.append(BubbleGroup(
                bbox=g.bbox,
                polygon=g.polygon,
                text=g.text,
                confidence=g.confidence,
                text_masks=(text_mask,),
                erase_masks=(erase_mask,),
                source=g.source,
                shape_kind=g.shape_kind,
                used_fallback=False,
                rotation_deg=g.rotation_deg,
                typesetting=g.typesetting,
                text_direction=g.text_direction,
            ))

        replaced = sum(1 for a, b in zip(augmented, groups) if a is not b)
        logger.debug("CTDAugmenter: %d/%d groups upgraded to pixel masks", replaced, len(groups))
        return tuple(augmented)

    def _get_backend(self):
        if self._backend is None:
            from typoon.vision._backends.ctd import CTDBackend
            logger.info("Loading CTD ONNX (mask augmenter)...")
            self._backend = CTDBackend(self._onnx_path)
        return self._backend


# ─── Matching ─────────────────────────────────────────────────────────────

def _best_match(lens_bbox, ctd_regions, used_indices: set[int] | None = None):
    """Return (CTD region, index) with highest IoU against lens_bbox, or (None, -1).

    Skips indices in ``used_indices`` so each CTD region is claimed by at
    most one Lens group — prevents the same pixel mask from being assigned
    to two overlapping groups when both beat the IoU threshold against the
    same large CTD region.
    """
    best_iou    = _IOU_MATCH_THRESHOLD
    best_region = None
    best_idx    = -1
    for i, region in enumerate(ctd_regions):
        if used_indices and i in used_indices:
            continue
        score = _iou(lens_bbox, region.bbox)
        if score > best_iou:
            best_iou    = score
            best_region = region
            best_idx    = i
    return best_region, best_idx


def _iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, aa + ab - inter)
