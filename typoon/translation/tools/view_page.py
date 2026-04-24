"""Page image encoder — no longer exposed as an LLM tool.

Kept for utilities that may need full-page encoding (e.g. CLI `detect`).
The translate module attaches bubble crops (view_bubble) by default,
not full pages.
"""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

_MAX_WIDTH = 1024
_MAX_PIXELS = 1_568 * 1_568
_JPEG_QUALITY = 40


def encode_page_jpeg(image: np.ndarray) -> str:
    """Encode RGB image to JPEG data URI, scaled for LLM vision input."""
    h, w = image.shape[:2]

    scale = min(1.0, _MAX_WIDTH / w)
    nw, nh = int(w * scale), int(h * scale)
    if nw * nh > _MAX_PIXELS:
        scale *= math.sqrt(_MAX_PIXELS / (nw * nh))
        nw, nh = int(w * scale), int(h * scale)

    if scale < 1.0:
        image = cv2.resize(image, (nw, nh))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
