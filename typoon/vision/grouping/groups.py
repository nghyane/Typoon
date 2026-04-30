"""Group building, OCR, filtering, erase-mask generation, and VTG export."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Protocol

import cv2
import numpy as np

from typoon.vision.types import (
    PageScanState,
    TextGroup,
    TextMask,
    VisualTextGroup,
)

from .geometry import (
    balance_fit_in_scope,
    box_iou,
    box_to_polygon,
    containment,
    expand,
    fit_padding,
    gx, gy, ox, oy,
    union_boxes,
)

_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")


class _Scanner(Protocol):
    _det: object
    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


# ── Group building ────────────────────────────────────────────────


def subgroup_text_blocks(
    indices: list[int],
    boxes: list[list[int]],
    container_box: list[int] | None = None,
) -> list[list[int]]:
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
        all_inside = all(
            (container_box[0] <= (boxes[i][0] + boxes[i][2]) / 2 <= container_box[2]
             and container_box[1] <= (boxes[i][1] + boxes[i][3]) / 2 <= container_box[3])
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
                same_col = ox(a, b) > 0.70 and gy(a, b) <= min_h * 0.45 and hr < 1.7
                same_row = oy(a, b) > 0.70 and gx(a, b) <= min_h * 1.10 and hr < 1.7
                overlap = False
            else:
                same_col = ox(a, b) > 0.55 and gy(a, b) <= min_h * 0.85 and hr < 2.1
                same_row = oy(a, b) > 0.60 and gx(a, b) <= min_h * 1.50 and hr < 2.1
                overlap = box_iou(a, b) > 0.12 or containment(a, b) > 0.35
            if same_col or same_row or overlap:
                join(i, j)

    groups_map: dict[int, list[int]] = {}
    for i in indices:
        groups_map.setdefault(find(i), []).append(i)
    return [sorted(g, key=lambda i: (boxes[i][1], boxes[i][0])) for g in groups_map.values()]


def _polygon_angle(polygon: list[list[float]]) -> float:
    if len(polygon) < 4:
        return 0.0
    bl, br = polygon[3], polygon[2]
    dx, dy = br[0] - bl[0], br[1] - bl[1]
    return math.degrees(math.atan2(dy, dx))


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def _cluster_by_angle(indices: list[int], angles: list[float], threshold: float = 20.0) -> list[list[int]]:
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
                cluster_angles[ci] = (ca * (len(clusters[ci]) - 1) + ang) / len(clusters[ci])
                placed = True
                break
        if not placed:
            clusters.append([i])
            cluster_angles.append(ang)
    return clusters


def _ocr_crop_box(
    group_box: list[int],
    group_indices: set[int],
    all_boxes: list[list[int]],
    page_w: int,
    page_h: int,
    scope_bbox: list[int] | None = None,
) -> list[int]:
    x1, y1, x2, y2 = group_box
    pad = int(max(x2 - x1, y2 - y1) * 0.10)
    left, top = max(0, x1 - pad), max(0, y1 - pad)
    right, bottom = min(page_w, x2 + pad), min(page_h, y2 + pad)

    if scope_bbox is not None:
        sx1, sy1, sx2, sy2 = scope_bbox
        lm, rm = x1 - sx1, sx2 - x2
        tm, bm = y1 - sy1, sy2 - y2
        if rm > lm: right = min(sx2, x2 + min(rm, max(pad, lm * 2)))
        if lm > rm: left = max(sx1, x1 - min(lm, max(pad, rm * 2)))
        if bm > tm: bottom = min(sy2, y2 + min(bm, max(pad, tm * 2)))
        if tm > bm: top = max(sy1, y1 - min(tm, max(pad, bm * 2)))

    for i, other in enumerate(all_boxes):
        if i in group_indices:
            continue
        ox1, oy1, ox2, oy2 = other
        vert_ovl = min(bottom, oy2) > max(top, oy1)
        horiz_ovl = min(right, ox2) > max(left, ox1)
        if vert_ovl:
            if ox2 <= x1 and ox2 > left: left = ox2 + 1
            elif ox1 >= x2 and ox1 < right: right = ox1 - 1
        if horiz_ovl:
            if oy2 <= y1 and oy2 > top: top = oy2 + 1
            elif oy1 >= y2 and oy1 < bottom: bottom = oy1 - 1

    return [min(left, x1), min(top, y1), max(right, x2), max(bottom, y2)]


def build_groups(state: PageScanState) -> None:
    active = [u.idx for u in state.units if not u.is_noise]
    boxes = [u.bbox for u in state.units]
    angles = [_polygon_angle(state.units[i].region.polygon) for i in range(len(state.units))]
    by_scope: dict[int, list[int]] = defaultdict(list)
    free: list[int] = []
    for i in active:
        scope_idx = state.units[i].scope_idx
        (free if scope_idx is None else by_scope[scope_idx]).append(i)

    raw_groups: list[tuple[list[int], bool, int | None]] = []

    for scope_idx, indices in by_scope.items():
        angle_clusters = _cluster_by_angle(indices, angles)
        if len(angle_clusters) == 1:
            for g in subgroup_text_blocks(indices, boxes, state.scopes[scope_idx].bbox):
                raw_groups.append((g, True, scope_idx))
        else:
            largest = max(angle_clusters, key=len)
            for cluster in angle_clusters:
                if cluster is largest:
                    for g in subgroup_text_blocks(cluster, boxes, state.scopes[scope_idx].bbox):
                        raw_groups.append((g, True, scope_idx))
                else:
                    free.extend(cluster)

    for cluster in _cluster_by_angle(free, angles):
        for g in subgroup_text_blocks(cluster, boxes, None):
            raw_groups.append((g, False, None))

    state.groups = []
    for gi, (indices, scoped, scope_idx) in enumerate(raw_groups):
        group_boxes = [boxes[i] for i in indices]
        raw = union_boxes(group_boxes)
        scope_bbox = state.scopes[scope_idx].bbox if scope_idx is not None else None
        ocr_box = _ocr_crop_box(raw, set(indices), boxes, state.width, state.height, scope_bbox)
        pad = fit_padding(group_boxes, state.width, state.height)
        fit = expand(raw, pad, state.width, state.height)
        if scope_bbox is not None:
            fit = [
                max(fit[0], scope_bbox[0]), max(fit[1], scope_bbox[1]),
                min(fit[2], scope_bbox[2]), min(fit[3], scope_bbox[3]),
            ]
            fit = balance_fit_in_scope(fit, scope_bbox, pad)
        med_angle = float(np.median([angles[i] for i in indices])) if indices else 0.0
        state.groups.append(TextGroup(gi, indices, scoped, scope_idx, raw, ocr_box, fit,
                                      scope_bbox=scope_bbox, median_angle=med_angle))


# ── OCR + filtering ───────────────────────────────────────────────


def ocr_groups(state: PageScanState, scanner: _Scanner) -> None:
    crops, groups = [], []
    for group in state.groups:
        x1, y1, x2, y2 = group.ocr_bbox
        if x2 - x1 >= 5 and y2 - y1 >= 5:
            crops.append(state.image[y1:y2, x1:x2])
            groups.append(group)
    for group, (text, conf) in zip(groups, scanner._ocr_crops(crops) if crops else []):
        group.ocr_text = (text or "").strip()
        group.ocr_conf = float(conf)


def final_filter_groups(state: PageScanState) -> None:
    for group in state.groups:
        skip, reason = _skip_final_group(group, state.width, state.height)
        group.accepted = not skip
        group.reject_reason = reason


def _is_uppercase_heavy(text: str) -> bool:
    alpha = sum(ch.isalpha() for ch in text)
    uppercase = sum(ch.isupper() for ch in text)
    return alpha >= 12 and uppercase / max(1, alpha) >= 0.55


def _looks_like_system_card(group: TextGroup, text: str, width_ratio: float, height_ratio: float) -> bool:
    if len(group.unit_indices) < 3:
        return False
    if width_ratio <= 0.24 and height_ratio <= 0.16:
        return False
    words = [p for p in re.split(r"\s+", text.strip()) if p]
    return len(words) >= 4 and _is_uppercase_heavy(text)


def _looks_like_narration(text: str, ocr_conf: float) -> bool:
    if ocr_conf < 0.70:
        return False
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if len(words) < 3:
        return False
    alnum = sum(ch.isalnum() for ch in text)
    if alnum < 10 or alnum / max(1, len(text)) < 0.55:
        return False
    return len(set(w.lower() for w in words)) >= min(3, len(words))


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


# ── Erase masks ───────────────────────────────────────────────────


def _dilate_text_mask(mask: TextMask, pad: int) -> TextMask:
    if pad <= 0:
        return mask
    mh, mw = mask.image.shape[:2]
    expanded = np.zeros((mh + pad * 2, mw + pad * 2), dtype=np.uint8)
    expanded[pad:pad + mh, pad:pad + mw] = mask.image
    ksize = pad * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return TextMask(x=mask.x - pad, y=mask.y - pad,
                    image=cv2.dilate(expanded, kernel, iterations=1))


def build_erase_masks(masks: list[TextMask], *, mode: str = "normal") -> list[TextMask]:
    out = []
    for mask in masks:
        h = mask.image.shape[0]
        pad = int(max(5, min(h * 0.16, 20))) if mode == "glow" else int(max(3, min(h * 0.10, 14)))
        out.append(_dilate_text_mask(mask, pad))
    return out


# ── VTG export ────────────────────────────────────────────────────


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
        text_masks = [
            state.units[i].region.mask
            for i in group.unit_indices
            if state.units[i].region.mask is not None
        ]
        mask_bbox = _mask_union_bbox(text_masks, group.fit_bbox)
        render_polygon = box_to_polygon(group.fit_bbox)
        erase_masks = build_erase_masks(
            text_masks, mode="glow" if _is_uppercase_heavy(group.ocr_text) else "normal"
        )
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
            scope_confidence=(
                state.scopes[group.scope_idx].confidence
                if group.scope_idx is not None else None
            ),
            text_masks=text_masks,
            erase_masks=erase_masks,
            source="scoped" if group.scoped else "free",
            unit_indices=list(group.unit_indices),
            accepted=group.accepted,
            reject_reason=group.reject_reason,
        ))
    return out
