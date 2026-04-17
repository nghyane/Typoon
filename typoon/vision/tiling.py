"""Shared vertical tiling for long images.

Both VisionScanner and PPocrScanner use this to handle manhwa/webtoon
strips that exceed their backend's effective resolution limit.
"""

from __future__ import annotations

import numpy as np

from .types import TextMask, TextRegion

_TILE_OVERLAP = 256


def compute_tiles(
    image_h: int, max_h: int, overlap: int = _TILE_OVERLAP,
) -> list[tuple[int, int]]:
    """Compute (y_offset, tile_height) pairs for vertical tiling."""
    if image_h <= max_h:
        return [(0, image_h)]

    tiles: list[tuple[int, int]] = []
    y = 0
    while y < image_h:
        th = min(max_h, image_h - y)
        tiles.append((y, th))
        if y + th >= image_h:
            break
        y += max_h - overlap
    return tiles


def offset_regions(
    regions: list[TextRegion],
    tile_y: int,
    full_image: np.ndarray,
) -> None:
    """Shift tile-local regions into full-image coordinates (in-place)."""
    h, w = full_image.shape[:2]
    for r in regions:
        r.polygon = [[p[0], p[1] + tile_y] for p in r.polygon]
        if r.mask is not None:
            r.mask = TextMask(x=r.mask.x, y=r.mask.y + tile_y, image=r.mask.image)
        x1 = max(0, int(min(p[0] for p in r.polygon)))
        y1 = max(0, int(min(p[1] for p in r.polygon)))
        x2 = min(w, int(max(p[0] for p in r.polygon)))
        y2 = min(h, int(max(p[1] for p in r.polygon)))
        r.crop = full_image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else r.crop


def deduplicate_regions(regions: list[TextRegion]) -> list[TextRegion]:
    """Remove near-duplicate regions from overlapping tiles (IoU > 0.5)."""
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
