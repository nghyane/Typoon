"""Lens proto → page-pixel geometry.

Two coordinate projections live here:

* ``norm_bbox`` — Lens normalised geometry (centre + size + rotation in
  the text box's own axes) → axis-aligned pixel bbox on the **page**.
  Used by both phases; tile pass adds a tile-local Y offset, bubble
  pass adds a crop origin + upscale factor.
* ``paragraph_to_raw`` — full Lens paragraph dict → :class:`RawBlock`,
  including per-line / per-word geometry.

Centralising the projection keeps the bubble re-OCR pass and the tile
pass in lock-step: a Lens block produced from a bubble crop has the
same shape as one produced from a tile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...contracts import LineBox, WordBox
from .types import RawBlock


__all__ = [
    "Frame",
    "norm_bbox",
    "paragraph_to_raw",
    "writing_direction",
]


# Lens proto writing_direction enum values.
_LENS_WD_LTR = 0
_LENS_WD_RTL = 1
_LENS_WD_T2B = 2

# Beyond this rotation magnitude an LTR/RTL paragraph reads as vertical
# on the page (text box's own axes ≠ page axes).
_VERTICAL_ROTATION_DEG = 45.0


@dataclass(frozen=True, slots=True)
class Frame:
    """Mapping from a Lens result's normalised space back to page pixels.

    Tile pass: ``origin_x=0``, ``origin_y=tile_top``, frame width/height
    = tile pixel dims, ``scale=1``.

    Bubble pass: origin = crop top-left in page pixels; frame width /
    height = crop pixel dims AFTER upscale; ``scale`` = upscale factor
    so we can divide back when projecting.
    """
    origin_x:    int
    origin_y:    int
    frame_w:     int
    frame_h:     int
    scale:       int = 1


def norm_bbox(
    geom: dict, frame: Frame, page_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Lens centre+size+rotation in [0, 1] → page-pixel AABB."""
    try:
        cx = float(geom["center_x"]) * frame.frame_w
        cy = float(geom["center_y"]) * frame.frame_h
        bw = float(geom["width"])    * frame.frame_w
        bh = float(geom["height"])   * frame.frame_h
    except (KeyError, TypeError, ValueError):
        return None
    angle_deg = float(geom.get("angle_deg") or 0.0)
    if abs(angle_deg) < 0.5:
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
    else:
        rad = math.radians(angle_deg)
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        hx, hy = bw / 2, bh / 2
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        xs = [cx + x * cos_t - y * sin_t for x, y in corners]
        ys = [cy + x * sin_t + y * cos_t for x, y in corners]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    # Divide-by-scale before adding origin: frame dims are crop pixels
    # AFTER upscale, but origin is in page pixels at native resolution.
    page_w, page_h = page_size
    px1 = max(0,          frame.origin_x + int(x1 / frame.scale))
    px2 = min(page_w,     frame.origin_x + int(x2 / frame.scale))
    py1 = max(0,          frame.origin_y + int(y1 / frame.scale))
    py2 = min(page_h,     frame.origin_y + int(y2 / frame.scale))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def writing_direction(raw_para, rotation_deg: float) -> str | None:
    """Lens proto direction → page-aligned ``vertical`` / ``horizontal``."""
    if raw_para is None:
        return None
    wd = getattr(raw_para, "writing_direction", None)
    if wd is None:
        return None
    if wd == _LENS_WD_T2B:
        return "vertical"
    if abs(rotation_deg) > _VERTICAL_ROTATION_DEG:
        return "vertical"
    return "horizontal"


def paragraph_to_raw(
    paragraph: dict,
    raw_paragraph,
    frame: Frame,
    page_size: tuple[int, int],
) -> RawBlock | None:
    """Parse one Lens paragraph dict into a :class:`RawBlock`."""
    text = (paragraph.get("text") or "").replace("\n", " ").strip()
    if not text:
        return None
    geom = paragraph.get("geometry") or {}
    bbox = norm_bbox(geom, frame, page_size)
    if bbox is None:
        return None

    words = tuple(
        WordBox(bbox=wb, text=wt)
        for wb, wt in _iter_words(paragraph, frame, page_size)
    )
    lines = tuple(
        LineBox(bbox=lb, text=lt, rotation_deg=lr)
        for lb, lt, lr in _iter_lines(paragraph, frame, page_size)
    )
    rotation = float(geom.get("angle_deg") or 0.0)
    direction = writing_direction(raw_paragraph, rotation) or "horizontal"

    return RawBlock(
        bbox=bbox, text=text, confidence=1.0, rotation_deg=rotation,
        words=words, lines=lines, text_direction=direction,
    )


def _iter_lines(paragraph: dict, frame: Frame, page_size):
    for line in paragraph.get("lines") or []:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        geom = line.get("geometry")
        if not geom:
            continue
        bbox = norm_bbox(geom, frame, page_size)
        if bbox is None:
            continue
        yield bbox, text, float(geom.get("angle_deg") or 0.0)


def _iter_words(paragraph: dict, frame: Frame, page_size):
    for line in paragraph.get("lines") or []:
        for word in line.get("words") or []:
            text = (word.get("text") or "").strip()
            if not text:
                continue
            geom = word.get("geometry")
            if not geom:
                continue
            bbox = norm_bbox(geom, frame, page_size)
            if bbox is None:
                continue
            yield bbox, text
