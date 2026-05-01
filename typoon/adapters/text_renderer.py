"""Minimal text renderer — fits translated text into a bubble fit box.

Font resolution order:
  1. Explicit path passed to render_text(font_path=...)
  2. Bundled typoon/assets/DejaVuSans.ttf (always present in package)

No system font paths. No fallback to PIL bitmap font (too small, unscalable).
"""

from __future__ import annotations

import importlib.resources
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=32)
def _get_font(size: int, font_path: str | None = None):
    from PIL import ImageFont

    if font_path:
        return ImageFont.truetype(font_path, size)

    # Bundled font — always available
    ref = importlib.resources.files("typoon.assets").joinpath("DejaVuSans.ttf")
    with importlib.resources.as_file(ref) as p:
        return ImageFont.truetype(str(p), size)


def _text_size(text: str, font) -> tuple[int, int]:
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def render_text(
    canvas: np.ndarray,
    text: str,
    fit_box: list[int],
    *,
    font_path: str | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
    max_font_size: int = 40,
    min_font_size: int = 8,
    padding: int = 4,
) -> tuple[int, bool]:
    """Draw translated text centered in fit_box on canvas (RGB or RGBA).

    Returns (font_size_used, overflow).
    overflow=True when text cannot fit even at min_font_size.
    """
    from PIL import Image, ImageDraw

    if not text or not text.strip():
        return 0, False

    x1, y1, x2, y2 = [int(v) for v in fit_box]
    box_w = max(1, x2 - x1 - padding * 2)
    box_h = max(1, y2 - y1 - padding * 2)

    # Binary search for largest font size where wrapped text fits
    lo, hi = min_font_size, max_font_size
    chosen_size = min_font_size
    chosen_lines: list[str] = [text]

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid, font_path)
        char_w, _ = _text_size("M", font)
        chars_per_line = max(1, box_w // max(1, char_w))
        lines = textwrap.wrap(text, width=chars_per_line) or [text]
        line_h = [_text_size(ln, font)[1] for ln in lines]
        total_h = sum(line_h) + (len(lines) - 1) * 2
        max_w = max(_text_size(ln, font)[0] for ln in lines)
        if max_w <= box_w and total_h <= box_h:
            chosen_size, chosen_lines = mid, lines
            lo = mid + 1
        else:
            hi = mid - 1

    font = _get_font(chosen_size, font_path)
    char_w, _ = _text_size("M", font)
    chars_per_line = max(1, box_w // max(1, char_w))
    chosen_lines = textwrap.wrap(text, width=chars_per_line) or [text]
    line_heights = [_text_size(ln, font)[1] for ln in chosen_lines]
    total_h = sum(line_heights) + (len(chosen_lines) - 1) * 2
    overflow = chosen_size == min_font_size and total_h > box_h

    # Render onto canvas region
    has_alpha = canvas.ndim == 3 and canvas.shape[2] == 4
    region = canvas[y1:y2, x1:x2]
    pil_img = Image.fromarray(region[:, :, :3] if has_alpha else region, "RGB")
    draw = ImageDraw.Draw(pil_img)

    y_cursor = (box_h - total_h) // 2 + padding
    for line, lh in zip(chosen_lines, line_heights):
        lw, _ = _text_size(line, font)
        x_pos = (box_w - lw) // 2 + padding
        draw.text((x_pos, y_cursor), line, font=font, fill=color)
        y_cursor += lh + 2

    rendered = np.array(pil_img)
    if has_alpha:
        canvas[y1:y2, x1:x2, :3] = rendered
    else:
        canvas[y1:y2, x1:x2] = rendered

    return chosen_size, overflow
