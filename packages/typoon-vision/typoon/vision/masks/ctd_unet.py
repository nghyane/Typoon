"""CtdUNetStrategy — crop refined CTD UNet bubble_mask per group bbox.

Pipeline:
  1. seg output (1,1,H,W float) → INTER_LINEAR resize → soft boundary
  2. threshold > 0.5 → binary
  3. morphClose(r=10) → fill holes inside bubble outline
  4. dilate(r=3) → expand slightly to cover balloon edge pixels
  5. clip to group.bbox → vùng ngoài bbox là noise (logo, art artifact),
     Lens filter đã xác nhận chỉ group bboxes là valid text regions

Cost: seg-only ONNX run (~265ms/page, CoreML) chạy 1 lần per page,
shared across all groups. Falls back to pixel_seg when unavailable.
"""

from __future__ import annotations

import cv2
import numpy as np

from typoon.vision.contracts import BubbleGroup, TextBlock, TextMask


__all__ = ["CtdUNetStrategy", "refine_seg"]

_HOLE_CLOSE_RADIUS = 10
_DILATION_RADIUS   = 3


def refine_seg(
    seg_raw: np.ndarray,    # (H, W) float, model output cropped to rw×rh
    orig_w: int,
    orig_h: int,
    rw: int,
    rh: int,
) -> np.ndarray:
    """seg output → refined uint8 (H, W) bubble mask.

    INTER_LINEAR resize preserves the soft UNet boundary.
    Close fills holes inside balloon outlines.
    Dilate expands to cover the balloon edge stroke.
    """
    crop = seg_raw[:rh, :rw]
    full = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary = (full > 0.5).astype(np.uint8) * 255

    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (_HOLE_CLOSE_RADIUS * 2 + 1,) * 2)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (_DILATION_RADIUS   * 2 + 1,) * 2)
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    return cv2.dilate(closed, k_dilate)


class CtdUNetStrategy:
    name = "ctd_unet"

    def build(
        self,
        group: BubbleGroup,
        members: tuple[TextBlock, ...],
        image: np.ndarray | None,
        bubble_mask: np.ndarray,          # refined uint8 (H, W), full page
    ) -> tuple[TextMask, ...]:
        """Clip refined bubble_mask to group bbox → 1 TextMask.

        Vùng ngoài group.bbox bị bỏ qua: Lens detection + filter đã
        xác nhận chỉ group bboxes là valid text regions; UNet activation
        ngoài bbox là noise (logo, art, panel border).
        """
        pH, pW = bubble_mask.shape[:2]
        x1, y1, x2, y2 = group.bbox
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(pW, x2), min(pH, y2)
        if x2c <= x1c or y2c <= y1c:
            return ()
        crop = bubble_mask[y1c:y2c, x1c:x2c].copy()
        if not (crop > 0).any():
            return ()
        return (TextMask(x=x1c, y=y1c, image=crop),)
