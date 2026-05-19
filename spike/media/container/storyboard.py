"""Storyboard builder — composite prepared JPEGs into a labeled grid.

No key overlay needed — the vision model receives bubble coordinates
in the text prompt. The image only provides visual context.

Input:  prepared/{chapter}/{i:04d}.jpg  (FUSE read)
Output: storyboard/{chapter}/{n:02d}.jpg
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_MAX_EDGE         = 2048
_LABEL_BAND       = 24   # px for page number banner
_LABEL_FONT_SIZE  = 18
PAGES_PER_CHUNK   = 9
JPEG_QUALITY      = 82


def chunk_pages(page_count: int) -> list[range]:
    return [
        range(start, min(start + PAGES_PER_CHUNK, page_count))
        for start in range(0, page_count, PAGES_PER_CHUNK)
    ]


def _grid_dims(n: int) -> tuple[int, int]:
    if n <= 4:  return 2, 2
    if n <= 6:  return 3, 2
    if n <= 9:  return 3, 3
    if n <= 12: return 4, 3
    cols = int(math.ceil(math.sqrt(n)))
    return cols, int(math.ceil(n / cols))


def _font(size: int) -> ImageFont.ImageFont:
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(cand).exists():
            try:
                return cast(ImageFont.ImageFont, ImageFont.truetype(cand, size))
            except OSError:
                pass
    return ImageFont.load_default()


def _build_cell(img: Image.Image, page_index: int, cell_w: int, cell_h: int) -> Image.Image:
    """Resize page to fit cell, add page number banner."""
    scale  = min(cell_w / img.width, cell_h / img.height, 1.0)
    scaled = img.resize((max(1, int(img.width * scale)),
                         max(1, int(img.height * scale))), Image.LANCZOS)

    out = Image.new("RGB", (scaled.width, scaled.height + _LABEL_BAND), (32, 32, 32))
    out.paste(scaled, (0, _LABEL_BAND))
    ImageDraw.Draw(out).text(
        (6, 3), f"p{page_index}",
        fill=(240, 240, 240), font=_font(_LABEL_FONT_SIZE),
    )
    return out


def _compose_grid(cells: list[Image.Image], cols: int, rows: int) -> Image.Image:
    pad = 4
    row_heights, col_widths = [], [0] * cols
    for r in range(rows):
        row = cells[r * cols:(r + 1) * cols]
        if not row: break
        row_heights.append(max(c.height for c in row))
        for ci, cell in enumerate(row):
            col_widths[ci] = max(col_widths[ci], cell.width)

    canvas = Image.new("RGB",
        (sum(col_widths) + pad * (cols + 1),
         sum(row_heights) + pad * (len(row_heights) + 1)),
        (20, 20, 20))
    y = pad
    for r, rh in enumerate(row_heights):
        x = pad
        for ci in range(cols):
            idx = r * cols + ci
            if idx >= len(cells): break
            canvas.paste(cells[idx], (x, y))
            x += col_widths[ci] + pad
        y += rh + pad

    longest = max(canvas.size)
    if longest > _MAX_EDGE:
        s = _MAX_EDGE / longest
        canvas = canvas.resize(
            (int(canvas.width * s), int(canvas.height * s)), Image.LANCZOS)
    return canvas


def build_storyboards(
    pages_rgb: dict[int, np.ndarray],
    page_order: list[int],
) -> list[tuple[range, bytes]]:
    """
    Build storyboard JPEGs from decoded page images.

    Args:
        pages_rgb:  {page_index → H×W×3 uint8}
        page_order: sorted list of page indices

    Returns:
        list of (page_range, jpeg_bytes) per chunk
    """
    chunks  = chunk_pages(len(page_order))
    results = []

    for chunk_range in chunks:
        n    = len(chunk_range)
        cols, rows = _grid_dims(n)
        cell_dim = {4: (900, 1200), 6: (800, 1100), 9: (700, 950)}
        cw, ch   = cell_dim.get(cols * rows, (600, 820))

        cells = []
        for local_i in chunk_range:
            pi  = page_order[local_i]
            img = Image.fromarray(pages_rgb[pi])
            cells.append(_build_cell(img, pi, cw, ch))

        grid = _compose_grid(cells, cols, rows)
        buf  = io.BytesIO()
        grid.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        results.append((chunk_range, buf.getvalue()))

    return results
