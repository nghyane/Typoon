"""Storyboard builder — composite prepared pages into a labeled canvas.

Each bubble is overlaid with its key on a red label so the vision agent
can map the OCR list back to bubbles in the image.

Layout strategy:
  1. Auto-detect ``n_cols`` = number of columns that fills ``_ROW_W_TARGET``
     without making cells too small, based on the median page aspect ratio.
  2. All cells have the SAME size: ``cell_w = row_w / n_cols``, ``cell_h``
     from the median aspect. No density-driven size difference — every page
     needs to be equally readable.
  3. Pages pack left-to-right in reading order; last row may be short.
     A short last row is NOT justify-stretched (avoids blowing up 1-page
     tails). Full rows are scaled to exactly ``_ROW_W_TARGET``.
  4. Bubble key labels scale with cell size. Font size is additionally
     boosted for pages with fewer, larger bubbles (so a single big speech
     bubble gets a more prominent label than a cluster of tiny SFX).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Layout config ─────────────────────────────────────────────────────────────

_ROW_W_TARGET      = 2048      # row width budget (px)
_CANVAS_MAX_EDGE   = 2048      # final JPEG longest edge cap
_MIN_CELL_H        = 300       # don't shrink cells below this
_MAX_CELL_H        = 1400      # don't let cells taller than this
_ROW_GAP           = 5
_CELL_GAP          = 4
_LABEL_BAND        = 20        # top band for page-index label
_LABEL_FONT_SIZE   = 15
_BUBBLE_LABEL_MIN  = 14
_BUBBLE_LABEL_MAX  = 28
_CELL_BG           = (30, 30, 30)
_CANVAS_BG         = (18, 18, 18)

STORYBOARD_JPEG_Q  = 82
PAGES_PER_CHUNK    = 9


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_pages(page_count: int) -> list[range]:
    return [
        range(start, min(start + PAGES_PER_CHUNK, page_count))
        for start in range(0, page_count, PAGES_PER_CHUNK)
    ]


# ── Font loading ──────────────────────────────────────────────────────────────

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}

def _font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if Path(cand).exists():
            try:
                f = cast(ImageFont.ImageFont, ImageFont.truetype(cand, size))
                _FONT_CACHE[size] = f
                return f
            except OSError:
                pass
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


# ── Page data ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Page:
    index:   int
    rgb:     np.ndarray
    bubbles: list[dict]   # [{"key": str, "bbox": [x1,y1,x2,y2]}]


# ── Auto column count ─────────────────────────────────────────────────────────

def _auto_cols(pages: list[_Page]) -> int:
    """Choose n_cols so cells are readable and rows fill the canvas width.

    Uses the median page aspect ratio (w/h). Tries cols 1..6 and picks the
    one where ``cell_h`` is closest to a comfortable reading height (500 px).
    """
    aspects = [p.rgb.shape[1] / max(1, p.rgb.shape[0]) for p in pages]
    med_aspect = median(aspects)

    target_h = 500      # comfortable reading height
    best_cols, best_score = 1, float("inf")
    for cols in range(1, 7):
        cell_w = (_ROW_W_TARGET - _CELL_GAP * (cols - 1)) / cols
        cell_h = cell_w / med_aspect
        score = abs(cell_h - target_h)
        if score < best_score:
            best_score = score
            best_cols  = cols
    return best_cols


# ── Cell sizing ───────────────────────────────────────────────────────────────

def _cell_dims(pages: list[_Page], n_cols: int) -> tuple[int, int]:
    """Return uniform (cell_w, cell_h) for all pages in this chunk.

    cell_w fills n_cols into _ROW_W_TARGET. cell_h derived from median aspect.
    """
    aspects = [p.rgb.shape[1] / max(1, p.rgb.shape[0]) for p in pages]
    med_aspect = median(aspects)
    cell_w = (_ROW_W_TARGET - _CELL_GAP * (n_cols - 1)) // n_cols
    cell_h = int(cell_w / med_aspect)
    cell_h = max(_MIN_CELL_H, min(_MAX_CELL_H, cell_h))
    return int(cell_w), cell_h


# ── Label font size ───────────────────────────────────────────────────────────

def _bubble_font_px(bubble_side_px: float) -> int:
    """Font size for a bubble label given the scaled bubble's shorter side."""
    return int(max(_BUBBLE_LABEL_MIN,
                   min(_BUBBLE_LABEL_MAX, bubble_side_px * 0.36)))


# ── Cell rendering ────────────────────────────────────────────────────────────

def _build_cell(page: _Page, cell_w: int, cell_h: int) -> Image.Image:
    """Render one page into a cell of exact size ``(cell_w, cell_h + _LABEL_BAND)``."""
    src_h, src_w = page.rgb.shape[:2]
    # Fit page preserving aspect ratio; fill remainder with cell bg.
    scale = min(cell_w / src_w, cell_h / src_h)
    rw = max(1, int(src_w * scale))
    rh = max(1, int(src_h * scale))
    scaled = Image.fromarray(page.rgb).resize((rw, rh), Image.LANCZOS)

    total_h = cell_h + _LABEL_BAND
    cell = Image.new("RGB", (cell_w, total_h), _CELL_BG)
    # Centre the scaled page within the cell area.
    ox = (cell_w - rw) // 2
    oy = _LABEL_BAND + (cell_h - rh) // 2
    cell.paste(scaled, (ox, oy))

    draw = ImageDraw.Draw(cell)
    draw.text((5, 2), f"p{page.index}",
              fill=(220, 220, 220), font=_font(_LABEL_FONT_SIZE))

    for b in page.bubbles:
        key  = b.get("key")
        bbox = b.get("bbox")
        if not key or not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        # Translate source coords → cell coords (centred paste + label band).
        cx = ox + (x1 + x2) * 0.5 * scale
        cy = oy + (y1 + y2) * 0.5 * scale
        bw = max(0, x2 - x1) * scale
        bh = max(0, y2 - y1) * scale
        font_px = _bubble_font_px(min(bw, bh))
        font    = _font(font_px)
        tb = draw.textbbox((0, 0), key, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(2, font_px // 5)
        x0 = int(cx - tw / 2 - pad)
        y0 = int(cy - th / 2 - pad)
        x3 = int(cx + tw / 2 + pad)
        y3 = int(cy + th / 2 + pad)
        draw.rectangle((x0, y0, x3, y3), fill=(220, 20, 20))
        draw.text((x0 + pad - tb[0], y0 + pad - tb[1]), key,
                  fill=(255, 255, 255), font=font)
    return cell


# ── Canvas composition ────────────────────────────────────────────────────────

def _compose_canvas(
    cells: list[Image.Image],
    n_cols: int,
) -> Image.Image:
    """Arrange cells left-to-right in rows of ``n_cols``.

    Full rows (exactly n_cols cells) are scaled to ``_ROW_W_TARGET`` so there
    is no trailing whitespace. The last (potentially short) row keeps its
    natural width and is left-aligned — no blowup on single-page tails.
    """
    n = len(cells)
    n_rows = -(-n // n_cols)   # ceil div
    cell_w, cell_h = cells[0].size

    # For full rows, scale factor to hit exact _ROW_W_TARGET.
    full_row_w = cell_w * n_cols + _CELL_GAP * (n_cols - 1)
    row_scale  = _ROW_W_TARGET / max(1, full_row_w)

    def scaled_cell(img: Image.Image, s: float) -> Image.Image:
        if abs(s - 1.0) < 0.01:
            return img
        return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)

    rows: list[list[Image.Image]] = []
    for r in range(n_rows):
        start = r * n_cols
        row_cells = cells[start : start + n_cols]
        is_full = len(row_cells) == n_cols
        s = row_scale if is_full else 1.0
        rows.append([scaled_cell(c, s) for c in row_cells])

    row_heights = [max(c.height for c in row) for row in rows]
    canvas_w = _ROW_W_TARGET + _CELL_GAP * 2
    canvas_h = sum(row_heights) + _ROW_GAP * (n_rows + 1)
    canvas   = Image.new("RGB", (canvas_w, canvas_h), _CANVAS_BG)
    y = _ROW_GAP
    for row, rh in zip(rows, row_heights):
        x = _CELL_GAP
        for cell in row:
            canvas.paste(cell, (x, y + (rh - cell.height) // 2))
            x += cell.width + _CELL_GAP
        y += rh + _ROW_GAP

    longest = max(canvas.size)
    if longest > _CANVAS_MAX_EDGE:
        s = _CANVAS_MAX_EDGE / longest
        canvas = canvas.resize(
            (int(canvas.width * s), int(canvas.height * s)), Image.LANCZOS)
    return canvas


# ── Entry point ───────────────────────────────────────────────────────────────

def build_storyboards(
    pages_rgb: dict[int, np.ndarray],
    page_order: list[int],
    bubbles_by_page: dict[int, list[dict]] | None = None,
) -> list[tuple[range, bytes]]:
    """Build storyboard JPEG chunks. Returns ``[(page_range, jpeg_bytes)]``.

    ``bubbles_by_page``: ``{page_index → [{"key", "bbox"}]}``. Each bubble's
    key is drawn as a red label on its bubble centre in source pixel coords.
    """
    bubbles_by_page = bubbles_by_page or {}
    chunks  = chunk_pages(len(page_order))
    results = []
    for chunk_range in chunks:
        pages = [
            _Page(
                index  = page_order[i],
                rgb    = pages_rgb[page_order[i]],
                bubbles= bubbles_by_page.get(page_order[i], []),
            )
            for i in chunk_range
        ]
        n_cols          = _auto_cols(pages)
        cell_w, cell_h  = _cell_dims(pages, n_cols)
        cells           = [_build_cell(p, cell_w, cell_h) for p in pages]
        canvas          = _compose_canvas(cells, n_cols)
        buf             = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=STORYBOARD_JPEG_Q, optimize=True)
        results.append((chunk_range, buf.getvalue()))
    return results
