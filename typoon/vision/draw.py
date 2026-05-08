"""Shared drawing helpers for vision debug overlays."""

from __future__ import annotations

import cv2
import numpy as np

CYAN = (0, 220, 255)
GREEN = (0, 220, 80)
RED = (255, 70, 70)
YELLOW = (255, 220, 0)
MAGENTA = (255, 0, 200)
WHITE = (255, 255, 255)

PALETTE = [
    (255, 80, 80), (80, 200, 255), (80, 255, 120), (255, 200, 0),
    (200, 80, 255), (255, 140, 0), (0, 220, 200), (255, 80, 180),
]


def rect(
    image: np.ndarray,
    box: list[int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def label(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    y = max(14, int(y))
    x = max(0, int(x))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.45, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 2), color, -1)
    cv2.putText(image, text, (x + 2, y - 2), font, scale, WHITE, thickness, cv2.LINE_AA)


def hstack(panels: list[np.ndarray]) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.full((h - p.shape[0], p.shape[1], 3), 240, dtype=np.uint8)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    return np.concatenate(padded, axis=1)


def write_rgb(path, image: np.ndarray) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")
