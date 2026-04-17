from __future__ import annotations

import base64

import cv2
import numpy as np
from pydantic import BaseModel, Field

from typoon.llm.tool_dec import tool

_MAX_WIDTH = 1024
_MAX_PIXELS = 1_568 * 1_568  # ~2.4M tokens budget
_JPEG_QUALITY = 40


class ViewPageArgs(BaseModel):
    page_index: int = Field(description="Zero-based page index")


@tool(strict=True)
async def view_page(args: ViewPageArgs) -> str:
    """View a comic page image for visual context.

    This is a SELECTIVE tool — only call when the OCR text is insufficient.
    If the page image is already attached in the user message, do not call again.

    When to use:
    - OCR text looks garbled, incomplete, or doesn't make sense in context.
    - Unnamed speaker, ambiguous age/rank not clear from dialogue.
    When NOT to use: all bubbles read clearly and speakers are obvious from text.
    """
    raise NotImplementedError("dispatch handles this")


def encode_page_jpeg(image: np.ndarray) -> str:
    """Encode RGB image to JPEG data URI.

    Scales to fit within _MAX_WIDTH and _MAX_PIXELS while keeping
    text readable on both regular manga and long manhwa strips.
    """
    h, w = image.shape[:2]
    import math

    # Scale to max width first (keeps text readable on long pages)
    scale = min(1.0, _MAX_WIDTH / w)
    # Then cap total pixels
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
