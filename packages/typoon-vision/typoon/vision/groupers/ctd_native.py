"""CTD-native grouper — 1 CTD text block = 1 BubbleGroup.

CTD already provides bubble-level bboxes from YoloV5 + per-block
UNet pixel masks, so no union-find merging is needed. Each TextBlock
becomes exactly one BubbleGroup:

  - erase_mask: UNet pixel mask (morphological close 3px) — tight,
    never bleeds onto neighbouring art unlike the dilated bbox approach.
  - text_mask:  same as erase_mask (no separate text region, CTD mask
    covers the glyph pixels).
  - text_direction: propagated from the detector's heuristic.
  - typesetting: None — CTD doesn't surface line geometry, so no
    FitHint is passed to render (pure binary-search baseline).
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from ..contracts import BubbleGroup, DetectionResult, TextMask


__all__ = ["CTDNativeGrouper"]

_CLOSE_RADIUS = 3   # morphological close on UNet mask to fill stroke gaps


class CTDNativeGrouper:
    """1 CTD TextBlock = 1 BubbleGroup. No merging."""

    name = "ctd_native"

    async def group(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        return await asyncio.to_thread(self._build, detection)

    def _build(self, detection: DetectionResult) -> tuple[BubbleGroup, ...]:
        groups: list[BubbleGroup] = []
        for block in detection.blocks:
            bbox_list = block.bbox

            if block.text_mask is not None:
                # UNet pixel mask — close small gaps in stroke outlines
                mask = _close_mask(block.text_mask)
            else:
                # Fallback: solid bbox rect (should not happen with CTD)
                mask = _bbox_mask(bbox_list)

            polygon = block.polygon or _bbox_polygon(bbox_list)

            groups.append(BubbleGroup(
                bbox=bbox_list,
                polygon=polygon,
                text="",                     # recognizer fills later
                confidence=block.confidence,
                text_masks=(mask,),
                erase_masks=(mask,),         # same mask — CTD is already tight
                source="ctd",
                shape_kind="dialogue",
                used_fallback=block.text_mask is None,
                rotation_deg=block.rotation_deg,
                typesetting=None,            # no Lens line geometry from CTD
                text_direction=block.text_direction,
            ))

        return tuple(groups)


# ─── Helpers ──────────────────────────────────────────────────────────────


def _close_mask(src: TextMask) -> TextMask:
    """Morphological close to fill stroke gaps, clipped to src bounds."""
    k = _CLOSE_RADIUS * 2 + 1
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed  = cv2.morphologyEx(src.image, cv2.MORPH_CLOSE, kernel)
    return TextMask(x=src.x, y=src.y, image=closed)


def _bbox_mask(bbox: tuple[int, int, int, int]) -> TextMask:
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    return TextMask(x=x1, y=y1, image=np.full((h, w), 255, dtype=np.uint8))


def _bbox_polygon(
    bbox: tuple[int, int, int, int],
) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = bbox
    return (
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    )
