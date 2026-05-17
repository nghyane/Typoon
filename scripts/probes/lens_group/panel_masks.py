"""Bottom-right panel: erase + text mask overlay.

  * Yellow halo — erase mask pixels (= filled container rect, the area
    AOT inpaint will redraw before render paints text).
  * Green       — text mask pixels (the tight per-member glyph extent).
  * Red contour — outline of the erase mask.

Erase and container share the same rectangle now, so the yellow band
visualises exactly where AOT writes inpaint into.
"""

from __future__ import annotations

import cv2
import numpy as np

from typoon.vision.contracts import BubbleGroup, TextMask

from .draw import PALETTE, draw_legend, label_panel


def render(
    canvas: np.ndarray,
    groups: tuple[BubbleGroup, ...],
    regions=(),  # unused; signature kept for __main__ compatibility
) -> np.ndarray:
    H, W = canvas.shape[:2]
    erase = np.zeros((H, W), dtype=np.uint8)
    text  = np.zeros((H, W), dtype=np.uint8)
    for g in groups:
        for em in g.erase_masks:
            _stamp(erase, em)
        for tm in g.text_masks:
            _stamp(text, tm)

    overlay = canvas.copy()
    erase_only = (erase > 0) & (text == 0)
    overlay[erase_only] = PALETTE["erase"]
    overlay[text > 0]   = PALETTE["text_mask"]
    blended = cv2.addWeighted(canvas, 0.45, overlay, 0.55, 0)

    contours, _ = cv2.findContours(erase, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (220, 0, 0), 1)

    draw_legend(blended, [
        ("erase (= container)", PALETTE["erase"]),
        ("glyph mask",          PALETTE["text_mask"]),
    ])
    label_panel(blended, "4. Erase (filled container) + glyph mask")
    return blended


def _stamp(target: np.ndarray, mask: TextMask) -> None:
    H, W = target.shape[:2]
    mx, my = mask.x, mask.y
    mh, mw = mask.image.shape[:2]
    x1, y1 = max(0, mx), max(0, my)
    x2, y2 = min(W, mx + mw), min(H, my + mh)
    if x2 <= x1 or y2 <= y1:
        return
    sx1, sy1 = x1 - mx, y1 - my
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    sub = mask.image[sy1:sy2, sx1:sx2]
    target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], sub)
