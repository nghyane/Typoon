"""2D tiling for PP-OCR text detection.

Square tiles preserve original resolution — no resize needed when tile fits
within the model's effective input size. Overlap larger than max bubble size
prevents bubbles from being split across tile boundaries.
"""

from __future__ import annotations

import numpy as np

from .types import TextMask, TextRegion

_TILE_SIZE = 1280      # tile width and height in pixels
_TILE_OVERLAP = 400    # must exceed max bubble dimension (~350px)


def compute_tiles_2d(
    image_h: int,
    image_w: int,
    tile_size: int = _TILE_SIZE,
    overlap: int = _TILE_OVERLAP,
) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) tile regions covering the full image with overlap."""
    stride = tile_size - overlap

    ys: list[int] = []
    y = 0
    while y < image_h:
        ys.append(y)
        if y + tile_size >= image_h:
            break
        y += stride

    xs: list[int] = []
    x = 0
    while x < image_w:
        xs.append(x)
        if x + tile_size >= image_w:
            break
        x += stride

    tiles: list[tuple[int, int, int, int]] = []
    for y in ys:
        th = min(tile_size, image_h - y)
        for x in xs:
            tw = min(tile_size, image_w - x)
            tiles.append((x, y, tw, th))
    return tiles


def offset_regions_2d(
    regions: list[TextRegion],
    tile_x: int,
    tile_y: int,
    full_image: np.ndarray,
) -> None:
    """Shift tile-local regions into full-image coordinates (in-place)."""
    h, w = full_image.shape[:2]
    for r in regions:
        r.polygon = [[p[0] + tile_x, p[1] + tile_y] for p in r.polygon]
        if r.mask is not None:
            r.mask = TextMask(
                x=r.mask.x + tile_x,
                y=r.mask.y + tile_y,
                image=r.mask.image,
            )
        x1 = max(0, int(min(p[0] for p in r.polygon)))
        y1 = max(0, int(min(p[1] for p in r.polygon)))
        x2 = min(w, int(max(p[0] for p in r.polygon)))
        y2 = min(h, int(max(p[1] for p in r.polygon)))
        r.crop = full_image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else r.crop


def deduplicate_regions(regions: list[TextRegion]) -> list[TextRegion]:
    """Remove near-duplicate regions from overlapping tiles.

    Two passes:
    1. IoU > 0.5 — standard overlap dedup (keeps higher confidence).
    2. Tile-seam fragment absorption — a small region that is x-adjacent
       (gap <= 4px) and y-overlapping (>= 50% of the smaller height) with a
       larger region is a split artifact; absorb it by extending the larger
       region's bbox and dropping the fragment.
    """
    if len(regions) <= 1:
        return regions

    def _bbox(poly):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        aa = (ax2 - ax1) * (ay2 - ay1)
        ab = (bx2 - bx1) * (by2 - by1)
        return inter / (aa + ab - inter)

    bboxes = [_bbox(r.polygon) for r in regions]
    keep = [True] * len(regions)

    # Pass 2: tile-seam fragment absorption — runs BEFORE IoU dedup so that
    # fragments are merged into their parent before the parent can be dropped.
    # A fragment is a region whose width is <= 30px AND <= 55% of the
    # adjacent region's width, x-touching (gap <= 4px), y-overlapping >= 50%.
    for i in range(len(regions)):
        if not keep[i]:
            continue
        bi = bboxes[i]
        iw = max(1, bi[2] - bi[0])
        ih = max(1, bi[3] - bi[1])
        if iw > 30:
            continue  # not a fragment candidate
        for j in range(len(regions)):
            if not keep[j] or i == j:
                continue
            bj = bboxes[j]
            jw = max(1, bj[2] - bj[0])
            jh = max(1, bj[3] - bj[1])
            if jw <= iw:
                continue  # j must be the larger region
            if iw > jw * 0.55:
                continue  # fragment must be meaningfully smaller
            # x-adjacent: gap <= 4px
            gap_x = max(0, max(bi[0], bj[0]) - min(bi[2], bj[2]))
            if gap_x > 4:
                continue
            # y-overlap >= 50% of smaller height
            y_inter = max(0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            if y_inter < min(ih, jh) * 0.50:
                continue
            # absorb fragment i into j by extending j's polygon
            new_x1 = min(bi[0], bj[0])
            new_x2 = max(bi[2], bj[2])
            new_y1 = min(bi[1], bj[1])
            new_y2 = max(bi[3], bj[3])
            regions[j].polygon = [
                [float(new_x1), float(new_y1)],
                [float(new_x2), float(new_y1)],
                [float(new_x2), float(new_y2)],
                [float(new_x1), float(new_y2)],
            ]
            bboxes[j] = (new_x1, new_y1, new_x2, new_y2)
            keep[i] = False
            break

    # Pass 1: IoU dedup
    for i in range(len(regions)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(regions)):
            if not keep[j]:
                continue
            if _iou(bboxes[i], bboxes[j]) > 0.5:
                if regions[i].confidence >= regions[j].confidence:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [r for r, k in zip(regions, keep) if k]


# Kept for backward compat — used by old vertical-only callers if any remain
def compute_tiles(
    image_h: int, max_h: int, overlap: int = 256,
) -> list[tuple[int, int]]:
    if image_h <= max_h:
        return [(0, image_h)]
    tiles: list[tuple[int, int]] = []
    y = 0
    while y < image_h:
        remaining = image_h - y
        if remaining <= max_h:
            tiles.append((y, remaining))
            break
        tiles.append((y, max_h))
        y += max_h - overlap
    return tiles


def offset_regions(
    regions: list[TextRegion],
    tile_y: int,
    full_image: np.ndarray,
) -> None:
    offset_regions_2d(regions, 0, tile_y, full_image)
