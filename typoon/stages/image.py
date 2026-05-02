"""Page image encoding with optional key overlay labels."""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

_MAX_WIDTH = 1024
_MAX_PIXELS = 1_568 * 1_568
_JPEG_QUALITY = 40

_LABEL_COLOR = (255, 80, 80)  # red in RGB
_LABEL_BG = (255, 255, 255)
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.45
_LABEL_THICK = 1


def encode_page_jpeg(
    image: np.ndarray,
    labels: dict[str, list[list[float]]] | None = None,
) -> str:
    """Encode RGB page image to JPEG data URI.

    If labels is provided, render key hash labels at each bubble's polygon
    centroid before encoding. This lets the vision model see which key
    corresponds to which bubble region.

    labels: {key: polygon} where polygon is [[x,y], ...].
    """
    img = image.copy() if labels else image

    if labels:
        for key, polygon in labels.items():
            _draw_label(img, key, polygon)

    h, w = img.shape[:2]
    scale = min(1.0, _MAX_WIDTH / w)
    nw, nh = int(w * scale), int(h * scale)
    if nw * nh > _MAX_PIXELS:
        scale *= math.sqrt(_MAX_PIXELS / (nw * nh))
        nw, nh = int(w * scale), int(h * scale)

    if scale < 1.0:
        img = cv2.resize(img, (nw, nh))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _draw_label(img: np.ndarray, key: str, polygon: list[list[float]]) -> None:
    """Draw a small key label near the top-center of the polygon."""
    if not polygon:
        return
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    cx = int(sum(xs) / len(xs))
    ty = int(min(ys)) - 4

    label = f"#{key}"
    (tw, th), _ = cv2.getTextSize(label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICK)
    lx = max(0, cx - tw // 2)
    ly = max(th + 2, ty)

    cv2.rectangle(img, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 2), _LABEL_BG, -1)
    cv2.putText(img, label, (lx, ly), _LABEL_FONT, _LABEL_SCALE, _LABEL_COLOR, _LABEL_THICK)
