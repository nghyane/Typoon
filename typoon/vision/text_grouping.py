"""Page-state based text grouping: PP-OCR → YOLO scope → subgroup → OCR.

YOLO is only a scope signal. Final FIT/erase regions are built from PP-OCR
text units and masks. The page state keeps stable unit/scope/group IDs for
runtime, preview, and debugging.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Protocol

import cv2
import numpy as np

from .tiling import compute_tiles, deduplicate_regions, offset_regions
from .types import (
    PageScanState,
    Scope,
    TextGroup,
    TextMask,
    TextRegion,
    TextUnit,
    VisualTextGroup,
)

PPOCR_MAX_TILE_HEIGHT = 2048
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")


class _GroupingScanner(Protocol):
    _det: object

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


# ── Geometry helpers ─────────────────────────────────────────────


def bbox(poly: list[list[float]]) -> list[int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def union_boxes(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


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


def _balance_fit_in_scope(fit: list[int], scope: list[int], pad: int) -> list[int]:
    """Expand fit_bbox toward the side that has more room inside the scope.

    Only fires when ALL three conditions hold for a given axis:
      1. imbalance >= 8 px  (skew is meaningful)
      2. smaller gap <= pad  (that side is genuinely tight against detected text;
                              if the gap is already larger than the normal padding
                              the text just isn't there — don't expand)
      3. larger gap > pad   (the other side actually has room worth filling)

    We expand only the tight side outward to match the roomy side, capped at
    the scope boundary.  We never shrink any side.
    """
    x1, y1, x2, y2 = fit
    sx1, sy1, sx2, sy2 = scope

    # Horizontal
    left_gap = x1 - sx1
    right_gap = sx2 - x2
    if (abs(left_gap - right_gap) >= 8
            and min(left_gap, right_gap) <= pad
            and max(left_gap, right_gap) > pad):
        target = max(left_gap, right_gap)
        x1 = max(sx1, x1 - max(0, target - left_gap))
        x2 = min(sx2, x2 + max(0, target - right_gap))

    # Vertical
    top_gap = y1 - sy1
    bottom_gap = sy2 - y2
    if (abs(top_gap - bottom_gap) >= 8
            and min(top_gap, bottom_gap) <= pad
            and max(top_gap, bottom_gap) > pad):
        target = max(top_gap, bottom_gap)
        y1 = max(sy1, y1 - max(0, target - top_gap))
        y2 = min(sy2, y2 + max(0, target - bottom_gap))

    return [x1, y1, x2, y2]


def _ox(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def _oy(a, b):
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def _gx(a, b):
    return max(0, max(a[0], b[0]) - min(a[2], b[2]))


def _gy(a, b):
    return max(0, max(a[1], b[1]) - min(a[3], b[3]))


def _box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, aa + bb - inter)


def _containment(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / min(max(1, (a[2] - a[0]) * (a[3] - a[1])),
                       max(1, (b[2] - b[0]) * (b[3] - b[1])))


# ── Detection and filtering ──────────────────────────────────────


def detect_raw_text_units(scanner: _GroupingScanner, image: np.ndarray) -> list[TextRegion]:
    from .tiling import compute_tiles_2d, offset_regions_2d
    h, w = image.shape[:2]
    all_regions: list[TextRegion] = []
    for tx, ty, tw, th in compute_tiles_2d(h, w):
        tile = image[ty:ty + th, tx:tx + tw]
        out = scanner._det.detect(tile)  # type: ignore[attr-defined]
        if tx or ty:
            offset_regions_2d(out.regions, tx, ty, image)
        all_regions.extend(out.regions)
    return deduplicate_regions(all_regions)


def unit_quality(unit: TextRegion, box: list[int], ocr_text: str, ocr_conf: float, *, filter_cjk: bool = False) -> tuple[bool, str | None]:
    w = max(1, box[2] - box[0])
    h = max(1, box[3] - box[1])
    area = w * h
    mask_area = int(np.count_nonzero(unit.mask.image)) if unit.mask is not None else 0
    if w < 4 or h < 4 or area < 24:
        return False, "tiny"
    if unit.confidence < 0.20 and mask_area < 12:
        return False, "low_det_low_mask"
    text = (ocr_text or "").strip()
    alnum = sum(ch.isalnum() for ch in text)
    if filter_cjk and CJK_RE.search(text):
        return False, "cjk_filtered"
    if not text and ocr_conf < 0.15 and unit.confidence < 0.45:
        return False, "ocr_empty_low_conf"
    if text and alnum == 0 and ocr_conf < 0.35:
        return False, "ocr_non_text_chars"
    return True, None


def detect_units(state: PageScanState, scanner: _GroupingScanner) -> None:
    regions = detect_raw_text_units(scanner, state.image)
    state.units = [TextUnit(idx=i, region=r, bbox=bbox(r.polygon)) for i, r in enumerate(regions)]


def ocr_units_for_filtering(state: PageScanState, scanner: _GroupingScanner) -> None:
    crops = [u.region.crop for u in state.units]
    ocr = scanner._ocr_crops(crops) if crops else []
    for u, (text, conf) in zip(state.units, ocr):
        u.unit_ocr_text = (text or "").strip()
        u.unit_ocr_conf = float(conf)


def filter_units(state: PageScanState) -> None:
    for u in state.units:
        ok, reason = unit_quality(u.region, u.bbox, u.unit_ocr_text, u.unit_ocr_conf)
        u.is_noise = not ok
        u.noise_reason = reason


_SPLIT_MIN_COVERAGE = 0.25  # each part must cover >= 25% of unit width


def split_units_crossing_scopes(state: PageScanState) -> None:
    """Split horizontal units that span across 2+ YOLO scope boundaries."""
    if not state.scopes:
        return
    if not state.units:
        return

    widths = [max(1, u.bbox[2] - u.bbox[0]) for u in state.units]
    median_w = float(np.median(widths)) if widths else 1.0

    new_units: list[TextUnit] = []
    idx = 0
    for u in state.units:
        ux1, uy1, ux2, uy2 = u.bbox
        uw = max(1, ux2 - ux1)

        # Only attempt split for wide units
        if uw < median_w * 1.5:
            u.idx = idx
            new_units.append(u)
            idx += 1
            continue

        # Find scopes with >= 25% horizontal coverage of this unit
        qualifying: list[tuple[int, int, int]] = []  # (scope_idx, overlap_x1, overlap_x2)
        for si, s in enumerate(state.scopes):
            sx1, _, sx2, _ = s.bbox
            ox1, ox2 = max(ux1, sx1), min(ux2, sx2)
            if ox2 <= ox1:
                continue
            if (ox2 - ox1) / uw >= _SPLIT_MIN_COVERAGE:
                qualifying.append((si, ox1, ox2))

        # Drop any scope that is fully contained within another qualifying scope —
        # nested scopes share the same text, splitting on them is incorrect.
        if len(qualifying) >= 2:
            def _x_range(q): return (state.scopes[q[0]].bbox[0], state.scopes[q[0]].bbox[2])
            filtered = []
            for qi in range(len(qualifying)):
                si = qualifying[qi][0]
                si_x1, _, si_x2, _ = state.scopes[si].bbox
                dominated = False
                for qj in range(len(qualifying)):
                    if qi == qj:
                        continue
                    sj_x1, _, sj_x2, _ = state.scopes[qualifying[qj][0]].bbox
                    if sj_x1 <= si_x1 and si_x2 <= sj_x2:
                        dominated = True
                        break
                if not dominated:
                    filtered.append(qualifying[qi])
            qualifying = filtered

        if len(qualifying) < 2:
            u.idx = idx
            new_units.append(u)
            idx += 1
            continue

        # Sort by x position and split at boundaries between consecutive scopes
        qualifying.sort(key=lambda t: t[1])
        crop = u.region.crop
        ch = crop.shape[0]

        prev_x2 = None
        for qi, (si, ox1, ox2) in enumerate(qualifying):
            if prev_x2 is None:
                sub_x1 = ux1
            else:
                sub_x1 = (prev_x2 + ox1) // 2
            sub_x2 = ox2 if qi == len(qualifying) - 1 else (ox2 + qualifying[qi + 1][1]) // 2
            sub_x2 = min(sub_x2, ux2)

            if sub_x2 - sub_x1 < 10:
                prev_x2 = ox2
                continue

            # Crop slice
            c1 = max(0, sub_x1 - ux1)
            c2 = min(crop.shape[1], sub_x2 - ux1)
            sub_crop = crop[:, c1:c2] if c2 > c1 else crop

            # Clip polygon
            sub_poly = [
                [min(max(float(p[0]), float(sub_x1)), float(sub_x2)), float(p[1])]
                for p in u.region.polygon
            ]

            # Clip mask
            sub_mask = None
            if u.region.mask is not None:
                m = u.region.mask
                mh, mw = m.image.shape[:2]
                mc1 = max(0, sub_x1 - m.x)
                mc2 = min(mw, sub_x2 - m.x)
                if mc2 > mc1:
                    sub_mask = TextMask(x=m.x + mc1, y=m.y, image=m.image[:, mc1:mc2].copy())

            from .types import TextRegion
            sub_region = TextRegion(
                polygon=sub_poly,
                crop=sub_crop,
                confidence=u.region.confidence,
                mask=sub_mask,
            )
            sub_bbox = [sub_x1, uy1, sub_x2, uy2]
            sub_unit = TextUnit(
                idx=idx,
                region=sub_region,
                bbox=sub_bbox,
                unit_ocr_text=u.unit_ocr_text,
                unit_ocr_conf=u.unit_ocr_conf,
                is_noise=u.is_noise,
                noise_reason=u.noise_reason,
            )
            new_units.append(sub_unit)
            idx += 1
            prev_x2 = ox2

    state.units = new_units


# ── Grouping ─────────────────────────────────────────────────────


def subgroup_text_blocks(indices: list[int], boxes: list[list[int]], container_box: list[int] | None = None) -> list[list[int]]:
    if not indices:
        return []
    if len(indices) == 1:
        return [indices]

    idx_boxes = [boxes[i] for i in indices]
    text_union = union_boxes(idx_boxes)
    heights = [max(1, b[3] - b[1]) for b in idx_boxes]
    med_h = float(np.median(heights))
    sorted_by_y = sorted(idx_boxes, key=lambda x: x[1])
    gaps = [max(0, sorted_by_y[k + 1][1] - sorted_by_y[k][3]) for k in range(len(sorted_by_y) - 1)]
    large_gaps = sum(1 for g in gaps if g > med_h * 1.25)

    mode = "strict" if container_box is None else "normal"
    if container_box is not None:
        cw = max(1, container_box[2] - container_box[0])
        ch = max(1, container_box[3] - container_box[1])
        union_w = max(1, text_union[2] - text_union[0])
        union_h = max(1, text_union[3] - text_union[1])
        n = len(indices)
        # All units inside scope → trust scope as boundary, keep together
        all_inside = all(
            (container_box[0] <= (boxes[i][0]+boxes[i][2])/2 <= container_box[2] and
             container_box[1] <= (boxes[i][1]+boxes[i][3])/2 <= container_box[3])
            for i in indices
        )
        if all_inside:
            return [list(indices)]
        compact = n <= 6 and union_h / ch < 0.85 and union_w / cw < 0.98 and large_gaps == 0
        if compact:
            return [list(indices)]
        if n >= 5 and (union_h / ch > 0.80 or large_gaps >= 2):
            mode = "strict"

    parent = {i: i for i in indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ni, i in enumerate(indices):
        a = boxes[i]
        ah = max(1, a[3] - a[1])
        for j in indices[ni + 1:]:
            b = boxes[j]
            bh = max(1, b[3] - b[1])
            min_h = max(1, min(ah, bh))
            hr = max(ah, bh) / min_h
            if mode == "strict":
                same_col = _ox(a, b) > 0.70 and _gy(a, b) <= min_h * 0.45 and hr < 1.7
                same_row = _oy(a, b) > 0.70 and _gx(a, b) <= min_h * 1.10 and hr < 1.7
                overlap = False
            else:
                same_col = _ox(a, b) > 0.55 and _gy(a, b) <= min_h * 0.85 and hr < 2.1
                same_row = _oy(a, b) > 0.60 and _gx(a, b) <= min_h * 1.50 and hr < 2.1
                overlap = _box_iou(a, b) > 0.12 or _containment(a, b) > 0.35
            if same_col or same_row or overlap:
                join(i, j)

    groups_map: dict[int, list[int]] = {}
    for i in indices:
        groups_map.setdefault(find(i), []).append(i)
    return [sorted(g, key=lambda i: (boxes[i][1], boxes[i][0])) for g in groups_map.values()]


def detect_scopes(state: PageScanState, yolo_model, yolo_imgsz: int, yolo_conf: float) -> None:
    if yolo_model is None:
        return
    from .bubble_scope import detect_bubble_scopes
    scopes = detect_bubble_scopes(yolo_model, state.image, imgsz=yolo_imgsz, conf=yolo_conf)
    state.scopes = [Scope(idx=i, bbox=s.bbox, confidence=s.confidence) for i, s in enumerate(scopes)]


def assign_units_to_scopes(state: PageScanState) -> None:
    if not state.scopes:
        return
    from .bubble_scope import BubbleScope, assign_units_to_scopes
    assignments = assign_units_to_scopes(
        [u.bbox for u in state.units],
        [BubbleScope(bbox=s.bbox, confidence=s.confidence) for s in state.scopes],
    )
    for u, scope_idx in zip(state.units, assignments):
        u.scope_idx = scope_idx


def _ocr_crop_box(group_box: list[int], group_indices: set[int], all_boxes: list[list[int]], page_w: int, page_h: int, scope_bbox: list[int] | None = None) -> list[int]:
    x1, y1, x2, y2 = group_box
    pad = int(max(x2 - x1, y2 - y1) * 0.10)
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(page_w, x2 + pad)
    bottom = min(page_h, y2 + pad)

    # Expand asymmetrically toward scope edges where text may be under-detected
    if scope_bbox is not None:
        sx1, sy1, sx2, sy2 = scope_bbox
        left_margin = x1 - sx1
        right_margin = sx2 - x2
        top_margin = y1 - sy1
        bottom_margin = sy2 - y2
        if right_margin > left_margin:
            right = min(sx2, x2 + min(right_margin, max(pad, left_margin * 2)))
        if left_margin > right_margin:
            left = max(sx1, x1 - min(left_margin, max(pad, right_margin * 2)))
        if bottom_margin > top_margin:
            bottom = min(sy2, y2 + min(bottom_margin, max(pad, top_margin * 2)))
        if top_margin > bottom_margin:
            top = max(sy1, y1 - min(top_margin, max(pad, bottom_margin * 2)))

    for i, other in enumerate(all_boxes):
        if i in group_indices:
            continue
        ox1, oy1, ox2, oy2 = other
        vertical_overlap = min(bottom, oy2) > max(top, oy1)
        horizontal_overlap = min(right, ox2) > max(left, ox1)
        if vertical_overlap:
            if ox2 <= x1 and ox2 > left:
                left = ox2 + 1
            elif ox1 >= x2 and ox1 < right:
                right = ox1 - 1
        if horizontal_overlap:
            if oy2 <= y1 and oy2 > top:
                top = oy2 + 1
            elif oy1 >= y2 and oy1 < bottom:
                bottom = oy1 - 1

    return [min(left, x1), min(top, y1), max(right, x2), max(bottom, y2)]


def _polygon_angle(polygon: list[list[float]]) -> float:
    """Angle of text line in degrees [-90, 90] from horizontal."""
    import math
    if len(polygon) < 4:
        return 0.0
    bl, br = polygon[3], polygon[2]
    dx = br[0] - bl[0]
    dy = br[1] - bl[1]
    return math.degrees(math.atan2(dy, dx))


def _cluster_by_angle(
    indices: list[int],
    angles: list[float],
    threshold: float = 20.0,
) -> list[list[int]]:
    """Split indices into clusters where angle difference <= threshold degrees."""
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    cluster_angles: list[float] = [angles[indices[0]]]
    for i in indices[1:]:
        ang = angles[i]
        placed = False
        for ci, ca in enumerate(cluster_angles):
            if _angle_diff(ang, ca) <= threshold:
                clusters[ci].append(i)
                # Update cluster representative angle (mean)
                cluster_angles[ci] = (ca * (len(clusters[ci]) - 1) + ang) / len(clusters[ci])
                placed = True
                break
        if not placed:
            clusters.append([i])
            cluster_angles.append(ang)
    return clusters


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def build_groups(state: PageScanState) -> None:
    active = [u.idx for u in state.units if not u.is_noise]
    boxes = [u.bbox for u in state.units]
    angles = [_polygon_angle(state.units[i].region.polygon) for i in range(len(state.units))]
    by_scope: dict[int, list[int]] = defaultdict(list)
    free: list[int] = []
    for i in active:
        scope_idx = state.units[i].scope_idx
        if scope_idx is None:
            free.append(i)
        else:
            by_scope[scope_idx].append(i)

    raw_groups: list[tuple[list[int], bool, int | None]] = []

    # For scoped groups: split by angle before subgrouping.
    # Units whose angle differs from the majority in the scope are moved to free.
    for scope_idx, indices in by_scope.items():
        angle_clusters = _cluster_by_angle(indices, angles)
        if len(angle_clusters) == 1:
            for g in subgroup_text_blocks(indices, boxes, state.scopes[scope_idx].bbox):
                raw_groups.append((g, True, scope_idx))
        else:
            # Keep the largest angle cluster in this scope, demote rest to free
            largest = max(angle_clusters, key=len)
            for cluster in angle_clusters:
                if cluster is largest:
                    for g in subgroup_text_blocks(cluster, boxes, state.scopes[scope_idx].bbox):
                        raw_groups.append((g, True, scope_idx))
                else:
                    free.extend(cluster)

    # For free groups: also split by angle
    angle_clusters = _cluster_by_angle(free, angles)
    for cluster in angle_clusters:
        for g in subgroup_text_blocks(cluster, boxes, None):
            raw_groups.append((g, False, None))

    state.groups = []
    for gi, (indices, scoped, scope_idx) in enumerate(raw_groups):
        group_boxes = [boxes[i] for i in indices]
        raw = union_boxes(group_boxes)
        scope_bbox = state.scopes[scope_idx].bbox if scope_idx is not None else None
        ocr = _ocr_crop_box(raw, set(indices), boxes, state.width, state.height, scope_bbox)
        pad = fit_padding(group_boxes, state.width, state.height)
        fit = expand(raw, pad, state.width, state.height)
        if scope_bbox is not None:
            fit = [
                max(fit[0], scope_bbox[0]),
                max(fit[1], scope_bbox[1]),
                min(fit[2], scope_bbox[2]),
                min(fit[3], scope_bbox[3]),
            ]
            fit = _balance_fit_in_scope(fit, scope_bbox, pad)
        group_angles = [angles[i] for i in indices]
        med_angle = float(np.median(group_angles)) if group_angles else 0.0
        state.groups.append(TextGroup(gi, indices, scoped, scope_idx, raw, ocr, fit, scope_bbox=scope_bbox, median_angle=med_angle))


def _is_uppercase_heavy(text: str) -> bool:
    alpha = sum(ch.isalpha() for ch in text)
    uppercase = sum(ch.isupper() for ch in text)
    return alpha >= 12 and uppercase / max(1, alpha) >= 0.55


def _looks_like_system_card(group: TextGroup, text: str, width_ratio: float, height_ratio: float) -> bool:
    if len(group.unit_indices) < 3:
        return False
    if width_ratio <= 0.24 and height_ratio <= 0.16:
        return False
    words = [part for part in re.split(r"\s+", text.strip()) if part]
    if len(words) < 4:
        return False
    return _is_uppercase_heavy(text)


def _looks_like_narration(text: str, ocr_conf: float) -> bool:
    """True for caption/narration text that is wide but clearly not SFX.

    Narration boxes outside bubbles are typically: high confidence, multiple
    distinct words, meaningful length, high alnum density.  SFX tend to be
    short repeated glyphs with low word variety.
    """
    if ocr_conf < 0.70:
        return False
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if len(words) < 3:
        return False
    alnum = sum(ch.isalnum() for ch in text)
    if alnum < 10:
        return False
    if alnum / max(1, len(text)) < 0.55:
        return False
    # SFX often repeat the same glyph — require enough distinct words
    distinct = len(set(w.lower() for w in words))
    return distinct >= min(3, len(words))


def _dilate_text_mask(mask: TextMask, pad: int) -> TextMask:
    if pad <= 0:
        return mask
    mh, mw = mask.image.shape[:2]
    expanded = np.zeros((mh + pad * 2, mw + pad * 2), dtype=np.uint8)
    expanded[pad:pad + mh, pad:pad + mw] = mask.image
    ksize = pad * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(expanded, kernel, iterations=1)
    return TextMask(x=mask.x - pad, y=mask.y - pad, image=dilated)


def build_erase_masks(masks: list[TextMask], *, mode: str = "normal") -> list[TextMask]:
    out: list[TextMask] = []
    for mask in masks:
        h = mask.image.shape[0]
        if mode == "glow":
            pad = int(max(5, min(h * 0.16, 20)))
        else:
            pad = int(max(3, min(h * 0.10, 14)))
        out.append(_dilate_text_mask(mask, pad))
    return out


# ── OCR/final filtering ──────────────────────────────────────────


def _crop_box(image: np.ndarray, box: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return image[y1:y2, x1:x2]


def _skip_final_group(group: TextGroup, page_w: int, page_h: int) -> tuple[bool, str | None]:
    text = group.ocr_text.strip()
    if not text:
        return True, "ocr_empty"
    bw = max(1, group.raw_bbox[2] - group.raw_bbox[0])
    bh = max(1, group.raw_bbox[3] - group.raw_bbox[1])
    area_ratio = (bw * bh) / max(1, page_w * page_h)
    width_ratio = bw / max(1, page_w)
    height_ratio = bh / max(1, page_h)
    alnum = sum(ch.isalnum() for ch in text)

    if alnum == 0:
        return True, "ocr_no_alnum"
    if not group.scoped:
        if _looks_like_system_card(group, text, width_ratio, height_ratio):
            return False, None
        if abs(group.median_angle) > 20.0:
            return True, "free_skewed"
        if _looks_like_narration(text, group.ocr_conf):
            return False, None
        if group.ocr_conf < 0.35:
            return True, "free_low_conf"
        if area_ratio > 0.025 or width_ratio > 0.24 or height_ratio > 0.16:
            return True, "free_large_sfx_like"
        if len(text) <= 2 and group.ocr_conf < 0.80:
            return True, "free_short_low_conf"
        if bw < 20 or bh < 20:
            return True, "free_tiny"
    return False, None


def ocr_groups(state: PageScanState, scanner: _GroupingScanner) -> None:
    crops: list[np.ndarray] = []
    groups: list[TextGroup] = []
    for group in state.groups:
        crop = _crop_box(state.image, group.ocr_bbox)
        if crop is not None:
            crops.append(crop)
            groups.append(group)
    for group, (text, conf) in zip(groups, scanner._ocr_crops(crops) if crops else []):
        group.ocr_text = (text or "").strip()
        group.ocr_conf = float(conf)


def final_filter_groups(state: PageScanState) -> None:
    for group in state.groups:
        skip, reason = _skip_final_group(group, state.width, state.height)
        group.accepted = not skip
        group.reject_reason = reason


def _mask_union_bbox(masks: list[TextMask], fallback: list[int]) -> list[int]:
    boxes = []
    for m in masks:
        mh, mw = m.image.shape[:2]
        boxes.append([int(m.x), int(m.y), int(m.x + mw), int(m.y + mh)])
    return union_boxes(boxes) if boxes else fallback


def to_visual_text_groups(state: PageScanState) -> list[VisualTextGroup]:
    out: list[VisualTextGroup] = []
    for group in state.groups:
        if not group.accepted:
            continue
        text_masks = [state.units[i].region.mask for i in group.unit_indices if state.units[i].region.mask is not None]
        mask_bbox = _mask_union_bbox(text_masks, group.fit_bbox)
        render_polygon = box_to_polygon(group.fit_bbox)
        erase_masks = build_erase_masks(text_masks, mode="glow" if _is_uppercase_heavy(group.ocr_text) else "normal")
        erase_bbox = _mask_union_bbox(erase_masks, group.fit_bbox)
        out.append(VisualTextGroup(
            text=group.ocr_text,
            confidence=group.ocr_conf,
            text_polygon=box_to_polygon(group.raw_bbox),
            render_polygon=render_polygon,
            text_bbox=group.raw_bbox,
            mask_bbox=mask_bbox,
            fit_bbox=group.fit_bbox,
            erase_bbox=erase_bbox,
            scope_bbox=group.scope_bbox,
            scope_confidence=(state.scopes[group.scope_idx].confidence if group.scope_idx is not None else None),
            text_masks=text_masks,
            erase_masks=erase_masks,
            source="scoped" if group.scoped else "free",
            unit_indices=list(group.unit_indices),
            accepted=group.accepted,
            reject_reason=group.reject_reason,
        ))
    return out


# ── Public API ───────────────────────────────────────────────────


def build_page_scan_state(
    scanner: _GroupingScanner,
    image: np.ndarray,
    *,
    yolo_model=None,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.3,
) -> PageScanState:
    h, w = image.shape[:2]
    state = PageScanState(image=image, width=w, height=h)
    detect_units(state, scanner)
    if not state.units:
        return state
    ocr_units_for_filtering(state, scanner)
    filter_units(state)
    detect_scopes(state, yolo_model, yolo_imgsz, yolo_conf)
    split_units_crossing_scopes(state)
    assign_units_to_scopes(state)
    build_groups(state)
    ocr_groups(state, scanner)
    final_filter_groups(state)
    return state


def group_and_ocr(
    scanner: _GroupingScanner,
    image: np.ndarray,
    *,
    yolo_model=None,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.3,
):
    state = build_page_scan_state(
        scanner, image,
        yolo_model=yolo_model,
        yolo_imgsz=yolo_imgsz,
        yolo_conf=yolo_conf,
    )
    return to_visual_text_groups(state)
