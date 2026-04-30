"""Geometry helpers shared across grouping pipeline."""

from __future__ import annotations

import numpy as np


def bbox(poly: list[list[float]]) -> list[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def union_boxes(boxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def expand(box: list[int], pad: int, w: int, h: int) -> list[int]:
    return [max(0, box[0] - pad), max(0, box[1] - pad),
            min(w, box[2] + pad), min(h, box[3] + pad)]


def box_to_polygon(box: list[int]) -> list[list[float]]:
    x1, y1, x2, y2 = box
    return [[float(x1), float(y1)], [float(x2), float(y1)],
            [float(x2), float(y2)], [float(x1), float(y2)]]


def fit_padding(boxes: list[list[int]], page_w: int, page_h: int) -> int:
    heights = [max(1, b[3] - b[1]) for b in boxes]
    med_h = float(np.median(heights)) if heights else 1.0
    if page_h / max(1, page_w) > 2.5:
        return int(max(4, min(med_h * 0.18, 18)))
    return int(max(2, min(med_h * 0.12, 10)))


def balance_fit_in_scope(fit: list[int], scope: list[int], pad: int) -> list[int]:
    """Expand fit_bbox toward the side with more room inside the scope."""
    x1, y1, x2, y2 = fit
    sx1, sy1, sx2, sy2 = scope

    left_gap, right_gap = x1 - sx1, sx2 - x2
    if (abs(left_gap - right_gap) >= 8
            and min(left_gap, right_gap) <= pad
            and max(left_gap, right_gap) > pad):
        target = max(left_gap, right_gap)
        x1 = max(sx1, x1 - max(0, target - left_gap))
        x2 = min(sx2, x2 + max(0, target - right_gap))

    top_gap, bottom_gap = y1 - sy1, sy2 - y2
    if (abs(top_gap - bottom_gap) >= 8
            and min(top_gap, bottom_gap) <= pad
            and max(top_gap, bottom_gap) > pad):
        target = max(top_gap, bottom_gap)
        y1 = max(sy1, y1 - max(0, target - top_gap))
        y2 = min(sy2, y2 + max(0, target - bottom_gap))

    return [x1, y1, x2, y2]


def ox(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def oy(a: list, b: list) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def gx(a: list, b: list) -> float:
    return max(0, max(a[0], b[0]) - min(a[2], b[2]))


def gy(a: list, b: list) -> float:
    return max(0, max(a[1], b[1]) - min(a[3], b[3]))


def box_iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, aa + bb - inter)


def containment(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / min(
        max(1, (a[2] - a[0]) * (a[3] - a[1])),
        max(1, (b[2] - b[0]) * (b[3] - b[1])),
    )
