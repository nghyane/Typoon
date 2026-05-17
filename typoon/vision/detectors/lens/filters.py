"""Block-level filters applied after Lens tile-pass dedup.

Drops decoration-only paragraphs, ridiculously small / hallucinated
huge bboxes, and cross-column union artefacts that Lens emits near
tile overlaps. Thresholds were tuned on the poc_lens_v3 fixture
survey and have stayed stable across redesigns.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from functools import cache

from ...contracts import TextBlock


__all__ = ["apply", "DECORATION_CHARS"]


# Size thresholds in page pixels.
_MIN_BBOX_W = 25
_MIN_BBOX_H = 18
_MIN_BBOX_AREA = 700
# Area/character beyond this is almost always a Lens hallucination on
# an art region.
_MAX_AREA_PER_CHAR = 6000

DECORATION_CHARS = frozenset("★☆●○◎◇◆□■▲△▽▼※・…—–-_=+×÷")

# Cross-column tile artefact: paragraph whose lines geometrically sit
# inside ≥ this many other paragraphs.
_CROSS_COLUMN_MIN_LINES_ABSORBED = 2
_CROSS_COLUMN_LINE_INSIDE_RATIO  = 0.70


def apply(
    blocks: list[TextBlock],
) -> tuple[list[TextBlock], list[tuple[TextBlock, str]]]:
    """Return (kept, rejected_with_reason)."""
    kept: list[TextBlock] = []
    rejected: list[tuple[TextBlock, str]] = []
    for b in blocks:
        text = b.text or ""
        if _bbox_too_small(b.bbox):
            rejected.append((b, "tiny_bbox"))
        elif _is_decoration_only(text):
            rejected.append((b, "decoration_only"))
        elif _bbox_too_large_for_text(b.bbox, text):
            rejected.append((b, "huge_bbox"))
        else:
            kept.append(b)

    kept, cross = _drop_cross_column_artifacts(kept)
    rejected.extend(cross)
    return kept, rejected


# ─── Per-block predicates ─────────────────────────────────────────────────


@cache
def _is_letter_or_digit(ch: str) -> bool:
    if ch in DECORATION_CHARS or ch.isspace():
        return False
    if ch.isalnum():
        return True
    return unicodedata.category(ch).startswith(("L", "N"))


def _is_decoration_only(text: str) -> bool:
    return not any(_is_letter_or_digit(c) for c in text)


def _bbox_too_small(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return w < _MIN_BBOX_W or h < _MIN_BBOX_H or w * h < _MIN_BBOX_AREA


def _bbox_too_large_for_text(bbox: tuple[int, int, int, int], text: str) -> bool:
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    chars = max(1, sum(1 for c in text if not c.isspace()))
    return area / chars > _MAX_AREA_PER_CHAR


# ─── Cross-column artefact ────────────────────────────────────────────────


def _drop_cross_column_artifacts(
    blocks: list[TextBlock],
) -> tuple[list[TextBlock], list[tuple[TextBlock, str]]]:
    """Drop paragraphs whose lines are absorbed by ≥2 other paragraphs.

    Lens occasionally stitches the tail lines of several adjacent
    tategaki columns into one phantom paragraph at tile overlaps.
    Each constituent line then geometrically sits inside a real column
    paragraph, so we drop the phantom by counting absorptions.
    """
    if len(blocks) < 3:
        return list(blocks), []

    kept: list[TextBlock] = []
    rejected: list[tuple[TextBlock, str]] = []
    for i, b in enumerate(blocks):
        if len(b.lines) < _CROSS_COLUMN_MIN_LINES_ABSORBED:
            kept.append(b)
            continue
        absorbing: set[int] = set()
        for ln in b.lines:
            for j, other in enumerate(blocks):
                if j == i:
                    continue
                if _bbox_inside_ratio(ln.bbox, other.bbox) >= _CROSS_COLUMN_LINE_INSIDE_RATIO:
                    absorbing.add(j)
                    break
        if len(absorbing) >= _CROSS_COLUMN_MIN_LINES_ABSORBED:
            rejected.append((b, "cross_column"))
        else:
            kept.append(b)
    return kept, rejected


def _bbox_inside_ratio(
    child: tuple[int, int, int, int], parent: tuple[int, int, int, int],
) -> float:
    cx1, cy1, cx2, cy2 = child
    px1, py1, px2, py2 = parent
    ix1, iy1 = max(cx1, px1), max(cy1, py1)
    ix2, iy2 = min(cx2, px2), min(cy2, py2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (cx2 - cx1) * (cy2 - cy1))
    return inter / area
