"""Top-right panel: Comic-DETR regions + empty-bubble flag."""

from __future__ import annotations

import cv2
import numpy as np

from typoon.vision.contracts import TextBlock

from .draw import PALETTE, draw_box, draw_legend, label_panel


def render(
    canvas: np.ndarray,
    regions: tuple[tuple[str, tuple[int, int, int, int], float], ...],
    blocks: list[TextBlock] | None = None,
) -> np.ndarray:
    """Draw DETR regions; flag bubbles that no Lens block landed inside."""
    out = canvas.copy()
    occupied = _occupied_bubbles(regions, blocks or [])
    for i, (cls, bbox, conf) in enumerate(regions):
        color = PALETTE.get(cls, (200, 200, 200))
        label = f"{cls} {conf:.2f}"
        if cls == "bubble" and i not in occupied:
            label += "  EMPTY"
            _stripe(out, bbox)
        draw_box(out, bbox, color, label=label, thickness=2)
    draw_legend(out, [
        ("bubble",      PALETTE["bubble"]),
        ("text_bubble", PALETTE["text_bubble"]),
        ("text_free",   PALETTE["text_free"]),
        ("empty bubble", (0, 0, 0)),
    ])
    label_panel(out, "2. Comic-DETR — bubble / text_bubble / text_free")
    return out


def _occupied_bubbles(regions, blocks: list[TextBlock]) -> set[int]:
    occ: set[int] = set()
    for j, (cls, bbox, _conf) in enumerate(regions):
        if cls != "bubble":
            continue
        for b in blocks:
            cx = (b.bbox[0] + b.bbox[2]) / 2.0
            cy = (b.bbox[1] + b.bbox[3]) / 2.0
            if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                occ.add(j)
                break
    return occ


def _stripe(canvas: np.ndarray, bbox) -> None:
    """Diagonal-stripe overlay to mark an empty bubble."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    stripe = canvas[y1:y2, x1:x2].copy()
    h, w = stripe.shape[:2]
    if h == 0 or w == 0:
        return
    overlay = np.zeros_like(stripe)
    for k in range(-h, w, 12):
        cv2.line(overlay, (k, 0), (k + h, h), (0, 0, 0), 1)
    canvas[y1:y2, x1:x2] = cv2.addWeighted(stripe, 0.7, overlay, 0.3, 0)
