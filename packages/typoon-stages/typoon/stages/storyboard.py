"""Storyboard — render N consecutive prepared pages into one labelled image.

The storyboard is the visual input fed to the chapter-context vision call.
Each bubble polygon center is overlaid with its 4-character key on a red
label so the model can address bubbles by ID without coordinate hand-waving.

Layout strategy is deterministic from page count:

  ≤4 pages   → 2×2 grid
  5–6 pages  → 3×2 grid
  7–9 pages  → 3×3 grid
  10–12 pages → 4×3 grid
  more       → chunked into multiple storyboards of 9 pages each

Cell size + max-edge are tuned from the speaker probe (Chainsaw Man ch.1):
9 pages in 3×3 at max_edge=2048 maintains readable bubble labels and gives
the vision model enough detail for stable speaker assignment (~89%
agreement with text-only baseline on the same data).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typoon.adapters.prepared_reader import PreparedReader
from typoon.domain.scan import BubbleKey

# Provider limits cap image side ~2048 in low-detail mode; going higher
# costs tokens without readability gain at this density.
_MAX_EDGE = 2048
_LABEL_BAND = 30
_LABEL_FONT_SIZE = 20
_KEY_LABEL_SIZE = 26
_KEY_PREFIX_CHARS = 4

# Per-storyboard page cap. 9 is empirically the sweet spot — beyond that
# the model's recall drops sharply (see speaker_probe_3x3 mixed-chapter
# result for the failure mode).
PAGES_PER_STORYBOARD = 9


@dataclass(frozen=True)
class StoryboardChunk:
    """One storyboard image covering a contiguous page range."""
    page_start: int          # inclusive
    page_end:   int          # exclusive
    image:      Image.Image  # rendered RGB image
    keys:       list[str]    # keys visible in this chunk, in reading order


def chunk_pages(page_count: int) -> list[range]:
    """Split [0, page_count) into chunks of up to PAGES_PER_STORYBOARD."""
    return [
        range(start, min(start + PAGES_PER_STORYBOARD, page_count))
        for start in range(0, page_count, PAGES_PER_STORYBOARD)
    ]


def _grid_dims(n: int) -> tuple[int, int]:
    """Pick (cols, rows) for n cells. Square-ish, slightly wider on ties."""
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 3, 2
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 4, 3
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return cols, rows


def _font(size: int) -> ImageFont.ImageFont:
    """System default font; fall back to PIL bitmap if missing."""
    for cand in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(cand).exists():
            try:
                return cast(ImageFont.ImageFont, ImageFont.truetype(cand, size))
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_key_labels(
    cell: Image.Image,
    scale: float,
    keyed_on_page: list[BubbleKey],
    font: ImageFont.ImageFont,
) -> None:
    """Overlay '#KEY' labels at each bubble polygon center on the cell.

    `scale` translates source-pixel coords to cell-pixel coords.
    """
    draw = ImageDraw.Draw(cell, "RGBA")
    for bk in keyed_on_page:
        poly = bk.bubble.polygon
        if not poly:
            continue
        xs = [p[0] * scale for p in poly]
        ys = [p[1] * scale for p in poly]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        label = bk.key[:_KEY_PREFIX_CHARS]
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 3
        box = (cx - tw / 2 - pad, cy - th / 2 - pad,
               cx + tw / 2 + pad, cy + th / 2 + pad)
        draw.rectangle(box, fill=(255, 80, 80, 235))
        draw.text((cx - tw / 2, cy - th / 2),
                  label, fill=(255, 255, 255), font=font)


def _build_cell(
    page_image: np.ndarray,
    keyed_on_page: list[BubbleKey],
    page_index: int,
    cell_w: int,
    cell_h: int,
) -> Image.Image:
    """Render one page as a labelled cell with a page-number banner."""
    raw = Image.fromarray(page_image)
    scale = min(cell_w / raw.width, cell_h / raw.height, 1.0)
    new_w = max(1, int(raw.width * scale))
    new_h = max(1, int(raw.height * scale))
    scaled = raw.resize((new_w, new_h), Image.LANCZOS)
    _draw_key_labels(scaled, scale, keyed_on_page, _font(_KEY_LABEL_SIZE))

    out = Image.new("RGB", (scaled.width, scaled.height + _LABEL_BAND), (32, 32, 32))
    out.paste(scaled, (0, _LABEL_BAND))
    d = ImageDraw.Draw(out)
    d.text((6, 3), f"page {page_index}", fill=(240, 240, 240),
           font=_font(_LABEL_FONT_SIZE))
    return out


def _compose_grid(cells: list[Image.Image], cols: int, rows: int) -> Image.Image:
    """Lay cells out in a grid with uniform column widths and row heights.

    Cells may have different sizes (manga vs splash page); we take per-row
    max height and per-column max width so the grid stays aligned without
    cropping any cell.
    """
    pad = 6
    row_heights: list[int] = []
    col_widths: list[int] = [0] * cols
    for r in range(rows):
        row = cells[r * cols:(r + 1) * cols]
        if not row:
            break
        row_heights.append(max(c.height for c in row))
        for ci, cell in enumerate(row):
            col_widths[ci] = max(col_widths[ci], cell.width)

    canvas_w = sum(col_widths) + pad * (cols + 1)
    canvas_h = sum(row_heights) + pad * (len(row_heights) + 1)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))

    y = pad
    for r, row_h in enumerate(row_heights):
        x = pad
        for ci in range(cols):
            idx = r * cols + ci
            if idx >= len(cells):
                break
            canvas.paste(cells[idx], (x, y))
            x += col_widths[ci] + pad
        y += row_h + pad

    longest = max(canvas.size)
    if longest > _MAX_EDGE:
        s = _MAX_EDGE / longest
        canvas = canvas.resize(
            (int(canvas.width * s), int(canvas.height * s)),
            Image.LANCZOS,
        )
    return canvas


def build_storyboard(
    reader: PreparedReader,
    keyed: list[BubbleKey],
    pages: range,
) -> StoryboardChunk:
    """Render `pages` as a single storyboard image.

    `keyed` is the full chapter key list; we filter to those whose
    bubble.page_index falls inside `pages`.
    """
    keys_by_page: dict[int, list[BubbleKey]] = {}
    for bk in keyed:
        if bk.bubble.page_index in pages:
            keys_by_page.setdefault(bk.bubble.page_index, []).append(bk)
    # Reading order within page
    for v in keys_by_page.values():
        v.sort(key=lambda bk: bk.bubble.idx)

    cols, rows = _grid_dims(len(pages))
    # Cell budget: enough room for a 1755×2500-class manga page after
    # accounting for max_edge cap. Empirically 700×950 at 3×3 keeps
    # labels readable; tighter grids can afford larger cells.
    if cols * rows <= 4:
        cell_w, cell_h = 900, 1200
    elif cols * rows <= 6:
        cell_w, cell_h = 800, 1100
    elif cols * rows <= 9:
        cell_w, cell_h = 700, 950
    else:
        cell_w, cell_h = 600, 820

    cells: list[Image.Image] = []
    for pi in pages:
        page_image = reader.read_rgb(pi)
        on_page = keys_by_page.get(pi, [])
        cells.append(_build_cell(page_image, on_page, pi, cell_w, cell_h))

    image = _compose_grid(cells, cols, rows)
    visible_keys: list[str] = []
    for pi in pages:
        for bk in keys_by_page.get(pi, []):
            visible_keys.append(bk.key)

    return StoryboardChunk(
        page_start=pages.start,
        page_end=pages.stop,
        image=image,
        keys=visible_keys,
    )


def encode_jpeg(image: Image.Image, quality: int = 88) -> bytes:
    """JPEG-encode the storyboard for vision provider transport."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
