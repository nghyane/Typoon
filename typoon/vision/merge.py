"""Merge detected text lines into bubbles by spatial proximity.

Groups PP-OCR text lines using angle, height, gap consistency,
and optionally a DB probability map for connected-component analysis.
Port of crates/engine/src/vision/merge.rs.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from .types import MergedBubble, TextRegion

# ── Thresholds ───────────────────────────────────────────────────────

_MIN_OVERLAP = 0.4
_MAX_ANGLE_DIFF = 10.0
_MAX_HEIGHT_RATIO = 1.8
_GAP_CONSISTENCY = 2.5

_PROB_CC_THRESH = 115  # ~0.45 in uint8
_PROB_GAP_RELAX = 1.25
_PROB_OVERLAP_RELAX = 0.7


# ═════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════


def group_lines(
    lines: list[TextRegion],
    prob_image: np.ndarray | None = None,
    vertical: bool = False,
    image: np.ndarray | None = None,
) -> list[MergedBubble]:
    """Group detected text lines into bubbles.

    Args:
        vertical: If True, merge vertical text columns (Japanese manga).
                  Transposes coordinates so the same horizontal algorithm works.
        image: Original page image (RGB uint8 HWC). When provided, enables
               BFS connectivity check that prevents merging across bubble borders.
    """
    if vertical:
        lines = [_transpose_region(r) for r in lines]
        if prob_image is not None:
            prob_image = prob_image.T
        if image is not None:
            image = np.ascontiguousarray(image.transpose(1, 0, 2))

    result = _group_lines_horizontal(lines, prob_image, image)

    if vertical:
        for b in result:
            b.polygon = _transpose_polygon(b.polygon)
            b.lines = [_transpose_region(r) for r in b.lines]
        # Japanese reading order: right-to-left columns
        result.sort(key=lambda b: -max(p[0] for p in b.polygon))

    return result


def _transpose_polygon(polygon: list[list[float]]) -> list[list[float]]:
    return [[y, x] for x, y in polygon]


def _transpose_region(r: TextRegion) -> TextRegion:
    return TextRegion(
        polygon=_transpose_polygon(r.polygon),
        crop=r.crop,
        confidence=r.confidence,
        mask=r.mask,
    )


def _group_lines_horizontal(
    lines: list[TextRegion],
    prob_image: np.ndarray | None = None,
    image: np.ndarray | None = None,
) -> list[MergedBubble]:
    """Core merge logic — assumes horizontal text lines."""
    if len(lines) <= 1:
        if not lines:
            return []
        r = lines[0]
        masks = [r.mask] if r.mask is not None else []
        return [MergedBubble(
            polygon=_bounding_polygon(lines), lines=lines,
            confidence=r.confidence, masks=masks,
        )]

    # Sort top-to-bottom
    lines = sorted(lines, key=lambda r: _top_y(r.polygon))

    stats = _PageStats.from_lines(lines)

    # Optional probability connected components
    comps = None
    if prob_image is not None:
        comps = _prob_components(prob_image)

    line_comps = [
        comps.assign(l.polygon, prob_image.shape[0]) if comps else None
        for l in lines
    ]

    groups: list[tuple[_Group, list[int]]] = []

    for i, line in enumerate(lines):
        lx1, ly1, lx2, ly2 = _bbox(line.polygon)
        lh = ly2 - ly1
        l_angle = _line_angle(line.polygon)
        l_comp = line_comps[i]

        best_gi: int | None = None
        best_vgap = float("inf")
        best_vgap_nearest = 0.0

        for gi, (g, _) in enumerate(groups):
            same = g.shares_component(l_comp)

            # Gate 1: orientation
            if abs(l_angle - g.avg_angle) > _MAX_ANGLE_DIFF:
                continue

            # Gate 2: font size (skip if vertically overlapping)
            v_overlap = min(g.y2, ly2) - max(g.y1, ly1)
            ratio = max(lh, g.avg_height) / max(min(lh, g.avg_height), 1.0)
            if ratio > _MAX_HEIGHT_RATIO and v_overlap / max(min(lh, g.y2 - g.y1), 1.0) < 0.3:
                continue

            # Gate 3: spatial proximity
            vgap = g.v_gap_to(ly1, ly2)
            hgap = g.h_gap_to(lx1, lx2)
            relax = _PROB_GAP_RELAX if same else 1.0
            if vgap > stats.max_v_gap * relax or hgap > stats.max_h_gap * relax:
                continue
            overlap_thresh = (_PROB_OVERLAP_RELAX if same else 1.0) * _MIN_OVERLAP
            if g.h_overlap_ratio(lx1, lx2) < overlap_thresh:
                continue

            nearest = g.nearest_bottom(ly1)
            vgap_n = max(ly1 - nearest, 0.0) if nearest is not None else vgap

            # Gate 4: gap consistency
            ref = g.median_gap
            if ref is None:
                ref = g.avg_height * (0.35 if g.count >= 2 else 1.0)
            ref = max(ref, g.avg_height * 0.1)
            if vgap_n > ref * _GAP_CONSISTENCY * relax:
                continue

            # Gate 5: BFS connectivity on original image
            if image is not None and vgap_n > 0:
                if not _are_connected(
                    image, nearest if nearest is not None else g.y2,
                    ly1, g.x1, g.x2, lx1, lx2,
                ):
                    continue

            if vgap < best_vgap:
                best_vgap = vgap
                best_vgap_nearest = vgap_n
                best_gi = gi

        if best_gi is not None:
            g, indices = groups[best_gi]
            g.add_line(line.polygon, best_vgap_nearest, l_comp)
            indices.append(i)
        else:
            groups.append((_Group(line.polygon, l_comp), [i]))

    # Build results
    result: list[MergedBubble] = []
    for _, indices in groups:
        group_lines_list = [lines[i] for i in indices]
        _sort_reading_order(group_lines_list)

        polygon = _bounding_polygon(group_lines_list)
        x1, y1, x2, y2 = _bbox(polygon)
        if (x2 - x1) < stats.min_bubble_w or (y2 - y1) < stats.min_bubble_h:
            continue

        result.append(MergedBubble(
            polygon=polygon,
            lines=group_lines_list,
            confidence=max(l.confidence for l in group_lines_list),
            masks=[l.mask for l in group_lines_list if l.mask is not None],
        ))

    result.sort(key=lambda b: _top_y(b.polygon))
    return result


# ═════════════════════════════════════════════════════════════════════
# Geometry
# ═════════════════════════════════════════════════════════════════════


def _bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _top_y(polygon: list[list[float]]) -> float:
    return min(p[1] for p in polygon)


def _line_angle(polygon: list[list[float]]) -> float:
    if len(polygon) >= 2:
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        a = abs(math.degrees(math.atan2(dy, dx)))
        return 180.0 - a if a > 90.0 else a
    return 0.0


def _bounding_polygon(regions: list[TextRegion]) -> list[list[float]]:
    x1 = min(p[0] for r in regions for p in r.polygon)
    y1 = min(p[1] for r in regions for p in r.polygon)
    x2 = max(p[0] for r in regions for p in r.polygon)
    y2 = max(p[1] for r in regions for p in r.polygon)
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# ═════════════════════════════════════════════════════════════════════
# Adaptive thresholds
# ═════════════════════════════════════════════════════════════════════


class _PageStats:
    __slots__ = ("max_v_gap", "max_h_gap", "min_bubble_w", "min_bubble_h")

    def __init__(self, median_h: float) -> None:
        self.max_v_gap = median_h * 1.5
        self.max_h_gap = median_h * 1.0
        self.min_bubble_w = median_h * 0.5
        self.min_bubble_h = median_h * 0.5

    @staticmethod
    def from_lines(lines: list[TextRegion]) -> _PageStats:
        heights = sorted(
            _bbox(l.polygon)[3] - _bbox(l.polygon)[1]
            for l in lines
            if _bbox(l.polygon)[3] - _bbox(l.polygon)[1] > 1.0
        )
        median_h = heights[len(heights) // 2] if heights else 20.0
        return _PageStats(median_h)


# ═════════════════════════════════════════════════════════════════════
# Probability connected components (uses cv2)
# ═════════════════════════════════════════════════════════════════════


class _ProbComponents:
    __slots__ = ("_labels", "_width")

    def __init__(self, labels: np.ndarray, width: int) -> None:
        self._labels = labels
        self._width = width

    def assign(self, polygon: list[list[float]], prob_h: int) -> int | None:
        x1, y1, x2, y2 = _bbox(polygon)
        x1i = min(int(x1), self._width - 1)
        y1i = min(int(y1), prob_h - 1)
        x2i = min(int(x2), self._width - 1)
        y2i = min(int(y2), prob_h - 1)

        counts: dict[int, int] = {}
        for sy in range(y1i, y2i + 1, 2):
            for sx in range(x1i, x2i + 1, 2):
                lbl = int(self._labels[sy, sx])
                if lbl != 0:
                    counts[lbl] = counts.get(lbl, 0) + 1

        if not counts:
            return None
        return max(counts, key=counts.get)  # type: ignore[arg-type]


def _prob_components(prob: np.ndarray) -> _ProbComponents:
    binary = (prob >= _PROB_CC_THRESH).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    return _ProbComponents(labels, prob.shape[1])


# ═════════════════════════════════════════════════════════════════════
# BFS connectivity check (Gate 5)
# ═════════════════════════════════════════════════════════════════════


def _pixel_lum(image: np.ndarray, x: int, y: int) -> int:
    """BT.601 luminance."""
    r, g, b = int(image[y, x, 0]), int(image[y, x, 1]), int(image[y, x, 2])
    return (r * 299 + g * 587 + b * 114) // 1000


def _otsu_threshold(image: np.ndarray, cx: int, cy: int, cw: int, ch: int,
                    scale: float, sw: int, sh: int) -> int:
    """Otsu's method on a crop region."""
    ih, iw = image.shape[:2]
    inv_scale = 1.0 / scale
    hist = np.zeros(256, dtype=np.int64)
    for sy in range(0, sh, 2):
        for sx in range(0, sw, 2):
            ox = cx + int(sx * inv_scale)
            oy = cy + int(sy * inv_scale)
            if ox < iw and oy < ih:
                lum = _pixel_lum(image, ox, oy)
                hist[min(lum, 255)] += 1
    total = int(hist.sum())
    if total == 0:
        return 128

    sum_all = float(np.dot(np.arange(256, dtype=np.float64), hist.astype(np.float64)))
    best_thresh = 0
    best_var = 0.0
    w0 = 0.0
    sum0 = 0.0
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * float(hist[t])
        mean0 = sum0 / w0
        mean1 = (sum_all - sum0) / w1
        var = w0 * w1 * (mean0 - mean1) ** 2
        if var > best_var:
            best_var = var
            best_thresh = t

    if best_var < 255.0 * 255.0 / 400.0:
        return 0
    return best_thresh


def _are_connected(
    image: np.ndarray,
    upper_y2: float, lower_y1: float,
    ax1: float, ax2: float,
    bx1: float, bx2: float,
) -> bool:
    """BFS on binarized luminance to check if two regions share a bubble interior."""
    ih, iw = image.shape[:2]

    if upper_y2 >= lower_y1:
        return True

    gap_h = lower_y1 - upper_y2
    pad = gap_h * 0.5
    cx1 = max(0, int(min(ax1, bx1) - pad))
    cy1 = max(0, int(upper_y2 - pad))
    cx2 = min(iw, int(max(ax2, bx2) + pad))
    cy2 = min(ih, int(lower_y1 + pad))
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw < 2 or ch < 2:
        return True

    scale = min(1.0, max(0.25, 4.0 / (min(cw, ch) * 0.03)))
    sw = max(2, int(math.ceil(cw * scale)))
    sh = max(2, int(math.ceil(ch * scale)))

    threshold = _otsu_threshold(image, cx1, cy1, cw, ch, scale, sw, sh)

    inv_scale = 1.0 / scale
    grid = np.zeros((sh, sw), dtype=np.bool_)
    for sy in range(sh):
        for sx in range(sw):
            ox = cx1 + int(sx * inv_scale)
            oy = cy1 + int(sy * inv_scale)
            if ox < iw and oy < ih:
                grid[sy, sx] = _pixel_lum(image, ox, oy) > threshold

    start_x = min(int(((ax1 + ax2) / 2 - cx1) * scale), sw - 1)
    start_y = min(int((upper_y2 - cy1) * scale), sh - 1)
    end_x = min(int(((bx1 + bx2) / 2 - cx1) * scale), sw - 1)
    end_y = min(int((lower_y1 - cy1) * scale), sh - 1)

    visited = np.zeros((sh, sw), dtype=np.bool_)
    queue: deque[tuple[int, int]] = deque()

    if grid[start_y, start_x]:
        visited[start_y, start_x] = True
        queue.append((start_x, start_y))
    else:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = start_x + dx, start_y + dy
                if 0 <= nx < sw and 0 <= ny < sh and grid[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        if not queue:
            return False

    ty_min = max(0, end_y - 2)
    ty_max = min(sh - 1, end_y + 2)
    tx_min = max(0, end_x - 2)
    tx_max = min(sw - 1, end_x + 2)

    while queue:
        cx, cy_pos = queue.popleft()
        if ty_min <= cy_pos <= ty_max and tx_min <= cx <= tx_max:
            return True
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy_pos + dy
            if 0 <= nx < sw and 0 <= ny < sh and grid[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((nx, ny))

    return False


# ═════════════════════════════════════════════════════════════════════
# Group state
# ═════════════════════════════════════════════════════════════════════


class _Group:
    __slots__ = (
        "x1", "y1", "x2", "y2",
        "_angle_sum", "_height_sum", "count",
        "_bottoms", "_gaps", "_comps",
    )

    def __init__(self, polygon: list[list[float]], comp_id: int | None) -> None:
        self.x1, self.y1, self.x2, self.y2 = _bbox(polygon)
        self._angle_sum = _line_angle(polygon)
        self._height_sum = self.y2 - self.y1
        self.count = 1
        self._bottoms = [self.y2]
        self._gaps: list[float] = []
        self._comps: list[int] = [comp_id] if comp_id is not None else []

    @property
    def avg_angle(self) -> float:
        return self._angle_sum / max(self.count, 1)

    @property
    def avg_height(self) -> float:
        return self._height_sum / max(self.count, 1)

    @property
    def median_gap(self) -> float | None:
        return self._gaps[len(self._gaps) // 2] if self._gaps else None

    def nearest_bottom(self, y: float) -> float | None:
        return min(self._bottoms, key=lambda b: abs(b - y)) if self._bottoms else None

    def shares_component(self, comp_id: int | None) -> bool:
        return comp_id is not None and comp_id in self._comps

    def v_gap_to(self, ly1: float, ly2: float) -> float:
        if ly2 < self.y1:
            return self.y1 - ly2
        if ly1 > self.y2:
            return ly1 - self.y2
        return 0.0

    def h_gap_to(self, lx1: float, lx2: float) -> float:
        if lx2 < self.x1:
            return self.x1 - lx2
        if lx1 > self.x2:
            return lx1 - self.x2
        return 0.0

    def h_overlap_ratio(self, lx1: float, lx2: float) -> float:
        overlap = min(self.x2, lx2) - max(self.x1, lx1)
        if overlap <= 0:
            return 0.0
        min_w = min(self.x2 - self.x1, lx2 - lx1)
        return overlap / min_w if min_w > 0 else 0.0

    def add_line(self, polygon: list[list[float]], v_gap: float, comp_id: int | None) -> None:
        x1, y1, x2, y2 = _bbox(polygon)
        self.x1, self.y1 = min(self.x1, x1), min(self.y1, y1)
        self.x2, self.y2 = max(self.x2, x2), max(self.y2, y2)
        self._angle_sum += _line_angle(polygon)
        self._height_sum += y2 - y1
        self.count += 1
        self._bottoms.append(y2)
        if comp_id is not None and comp_id not in self._comps:
            self._comps.append(comp_id)
        min_meaningful = self.avg_height * 0.03
        if v_gap > min_meaningful:
            self._gaps.append(v_gap)
            self._gaps.sort()


# ═════════════════════════════════════════════════════════════════════
# Reading order
# ═════════════════════════════════════════════════════════════════════


def _sort_reading_order(lines: list[TextRegion]) -> None:
    if len(lines) <= 1:
        return

    metas = []
    for l in lines:
        x1, y1, _, y2 = _bbox(l.polygon)
        metas.append((((y1 + y2) / 2), y2 - y1, x1))

    max_h = max(m[1] for m in metas)
    tol = max_h * 0.35

    indices = sorted(range(len(lines)), key=lambda i: metas[i][0])

    rows: list[list[int]] = []
    row = [indices[0]]

    for idx in indices[1:]:
        row_cy = sum(metas[i][0] for i in row) / len(row)
        if abs(metas[idx][0] - row_cy) <= tol:
            row.append(idx)
        else:
            rows.append(row)
            row = [idx]
    rows.append(row)

    for r in rows:
        r.sort(key=lambda i: metas[i][2])

    ordered = [lines[i] for r in rows for i in r]
    lines[:] = ordered
