from __future__ import annotations

import base64
import math

import cv2
import numpy as np
from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool

_MAX_DIM = 384
_JPEG_QUALITY = 85
_CONTEXT_PAD = 0.3  # 30% padding around bubble bbox


class ViewBubbleArgs(BaseModel):
    bubble_id: str = Field(description="Bubble ID, e.g. 'p0_b3'")


@tool(strict=True)
async def view_bubble(args: ViewBubbleArgs) -> str:
    """View a single bubble's region at full resolution.

    Use when OCR text for a specific bubble is garbled or unclear.
    Returns a cropped image of the bubble with context padding.
    Prefer this over view_page when you only need to verify one bubble.
    """
    raise NotImplementedError("dispatch handles this")


def encode_bubble_jpeg(image: np.ndarray, polygon: list[list[float]]) -> str:
    """Crop bubble region with padding, encode as JPEG data URI."""
    h, w = image.shape[:2]

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    pad_x = int(bw * _CONTEXT_PAD)
    pad_y = int(bh * _CONTEXT_PAD)

    x1 = max(0, int(math.floor(min(xs))) - pad_x)
    y1 = max(0, int(math.floor(min(ys))) - pad_y)
    x2 = min(w, int(math.ceil(max(xs))) + pad_x)
    y2 = min(h, int(math.ceil(max(ys))) + pad_y)

    crop = image[y1:y2, x1:x2]

    ch, cw = crop.shape[:2]
    if max(ch, cw) > _MAX_DIM:
        scale = _MAX_DIM / max(ch, cw)
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)))

    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
