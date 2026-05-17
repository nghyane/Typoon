"""Reusable drawing primitives — colour palette + box/legend helpers."""

from __future__ import annotations

import cv2
import numpy as np


# Shared RGB palette across panels.
PALETTE: dict[str, tuple[int, int, int]] = {
    "word":        (255, 200,   0),   # gold
    "line":        (  0, 200, 255),   # cyan
    "paragraph":   (255,   0, 255),   # magenta
    "bubble":      (255,  60,  60),   # red
    "text_bubble": ( 60, 200,  60),   # green
    "text_free":   ( 60,  60, 255),   # blue
    "container":   (255,   0,   0),   # red — final render polygon
    "erase":       (255, 230,   0),   # yellow halo
    "text_mask":   (  0, 255,   0),   # green
}


def draw_box(
    canvas: np.ndarray, bbox, color: tuple[int, int, int],
    label: str = "", *, thickness: int = 2, font_scale: float = 0.4,
) -> None:
    if bbox is None:
        return
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            canvas, label, (x1 + 2, max(12, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
        )


def draw_legend(canvas: np.ndarray, entries: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 8
    for label, color in entries:
        cv2.rectangle(canvas, (8, y), (24, y + 12), color, -1)
        cv2.putText(
            canvas, label, (30, y + 11),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA,
        )
        y += 16


def label_panel(canvas: np.ndarray, title: str) -> None:
    """Stamp a title bar at the top of the panel."""
    H = canvas.shape[0]
    bar_h = max(22, H // 40)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], bar_h), (32, 32, 32), -1)
    cv2.putText(
        canvas, title, (8, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
