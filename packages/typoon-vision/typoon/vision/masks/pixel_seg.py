"""PixelSegStrategy — dialogue bubble mask via pixel-based segmentation.

Seeds from word bbox pixels only, runs 3-step morphological close,
clips hard to word_union boundary, smooths edges with Gaussian blur.
Falls back to rect-dilate when image is unavailable.
"""

from __future__ import annotations

import statistics

import cv2
import numpy as np

from typoon.vision.contracts import BubbleGroup, TextBlock, TextMask
from typoon.vision.groupers._spatial_join import _median_glyph_size


__all__ = ["PixelSegStrategy", "RectDilateStrategy"]

_CLIP_PAD = 4   # px margin around word_union for dilation room


class PixelSegStrategy:
    name = "pixel_seg"

    def build(
        self,
        group: BubbleGroup,
        members: tuple[TextBlock, ...],
        image: np.ndarray | None,
    ) -> tuple[TextMask, ...]:
        word_boxes = [w.bbox for m in members for w in m.words] \
                     or [m.bbox for m in members]

        wu_x1 = min(b[0] for b in word_boxes)
        wu_y1 = min(b[1] for b in word_boxes)
        wu_x2 = max(b[2] for b in word_boxes)
        wu_y2 = max(b[3] for b in word_boxes)

        glyph = _median_glyph_size(list(members))
        # Bridge radius: glyph short-side, NOT line height.
        # For tategaki, line height = column length (can be 60-100px) —
        # using it as close radius floods a 60px-wide ROI entirely.
        bridge = max(glyph, 8)

        if image is not None:
            return self._pixel_seg(image, word_boxes, wu_x1, wu_y1, wu_x2, wu_y2, glyph, bridge)

        return RectDilateStrategy().build(group, members, image)

    def _pixel_seg(
        self,
        image: np.ndarray,
        word_boxes: list,
        wu_x1: int, wu_y1: int, wu_x2: int, wu_y2: int,
        glyph: int, bridge: int,
    ) -> tuple[TextMask, ...]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) \
               if image.ndim == 3 else image
        pH, pW = gray.shape

        # Seed: Otsu threshold per word bbox — adapts to local ink/bg contrast.
        # Per-bbox Otsu handles both thin Japanese glyphs (low ink pixel count)
        # and bold English lettering without a fixed threshold cap.
        seed = np.zeros((pH, pW), dtype=np.uint8)
        for wb in word_boxes:
            wx1, wy1 = max(0, wb[0]), max(0, wb[1])
            wx2, wy2 = min(pW, wb[2]), min(pH, wb[3])
            patch = gray[wy1:wy2, wx1:wx2]
            if patch.size < 9:
                continue
            _, bp = cv2.threshold(
                patch, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            )
            seed[wy1:wy2, wx1:wx2] = bp

        # Crop to word_union + CLIP_PAD
        rx1 = max(0, wu_x1 - _CLIP_PAD)
        ry1 = max(0, wu_y1 - _CLIP_PAD)
        rx2 = min(pW, wu_x2 + _CLIP_PAD)
        ry2 = min(pH, wu_y2 + _CLIP_PAD)
        blob = seed[ry1:ry2, rx1:rx2].copy()
        roi_h, roi_w = blob.shape[:2]

        # Close kernel cap: separate per axis so tategaki narrow columns
        # (roi_w ≈ glyph ≈ 14px) don't get flooded horizontally, while
        # vertical bridging can still span the inter-word gap (~7px).
        # cap_x = roi_w // 3  → keeps horizontal spread inside the column
        # cap_y = roi_h // 3  → allows vertical bridging across word gaps
        # Final radius = min(base, cap_x, cap_y) for a circular kernel.
        cap = min(roi_w // 3, roi_h // 3)
        cap = max(cap, 2)

        def _r(base: int) -> int:
            return min(base, cap)

        for r in (_r(max(2, glyph // 4)), _r(max(4, int(glyph * 0.8))), _r(max(6, int(bridge * 0.7)))):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
            blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, k)

        # Hard-clip to word_union
        cm = np.zeros_like(blob)
        cy1_ = wu_y1 - ry1
        cy2_ = wu_y2 - ry1 + _CLIP_PAD * 2
        cx1_ = wu_x1 - rx1
        cx2_ = wu_x2 - rx1 + _CLIP_PAD * 2
        cm[max(0, cy1_):min(blob.shape[0], cy2_),
           max(0, cx1_):min(blob.shape[1], cx2_)] = 255
        blob = cv2.bitwise_and(blob, cm)

        # Keep largest CC
        n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            blob = (labels == largest).astype(np.uint8) * 255

        # Dilate + Gaussian blur → smooth edges like UNet soft boundary
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blob = cv2.dilate(blob, k_dil, iterations=1)
        blob = cv2.GaussianBlur(blob, (7, 7), sigmaX=2.0)
        _, blob = cv2.threshold(blob, 80, 255, cv2.THRESH_BINARY)

        return (TextMask(x=rx1, y=ry1, image=blob),)


class RectDilateStrategy:
    """Fallback when no image — word union rect + dilate."""

    name = "rect_dilate"

    def build(
        self,
        group: BubbleGroup,
        members: tuple[TextBlock, ...],
        image: np.ndarray | None,
    ) -> tuple[TextMask, ...]:
        word_boxes = [w.bbox for m in members for w in m.words] \
                     or [m.bbox for m in members]

        wu_x1 = min(b[0] for b in word_boxes)
        wu_y1 = min(b[1] for b in word_boxes)
        wu_x2 = max(b[2] for b in word_boxes)
        wu_y2 = max(b[3] for b in word_boxes)

        glyph = _median_glyph_size(list(members))
        line_h_list = [
            ln.bbox[3] - ln.bbox[1]
            for m in members for ln in m.lines
            if ln.bbox[3] - ln.bbox[1] > 0
        ]
        lh = int(statistics.median(line_h_list)) if line_h_list else max(glyph, 8)

        margin = 1
        W = max(1, wu_x2 - wu_x1 + margin * 2)
        H = max(1, wu_y2 - wu_y1 + margin * 2)
        canvas = np.zeros((H, W), dtype=np.uint8)
        for b in word_boxes:
            cv2.rectangle(
                canvas,
                (b[0] - wu_x1 + margin, b[1] - wu_y1 + margin),
                (b[2] - wu_x1 + margin, b[3] - wu_y1 + margin),
                255, -1,
            )
        radius = max(2, int(lh * 0.9))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        blob = cv2.dilate(canvas, k, iterations=1)
        return (TextMask(x=wu_x1 - margin, y=wu_y1 - margin, image=blob),)
