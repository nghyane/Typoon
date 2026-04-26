"""Page-state based text grouping: PP-OCR → YOLO scope → subgroup → OCR.

YOLO is only a scope signal. Final FIT/erase regions are built from PP-OCR
text units and masks. The page state keeps stable unit/scope/group IDs for
runtime, preview, and debugging.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Protocol

import cv2
import numpy as np

from .tiling import compute_tiles, deduplicate_regions, offset_regions
from .types import TextMask, TextRegion, VisualTextGroup

PPOCR_MAX_TILE_HEIGHT = 2048
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")


class _GroupingScanner(Protocol):
    _det: object

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


@dataclass
class TextUnit:
    idx: int
    region: TextRegion
    bbox: list[int]
    unit_ocr_text: str = ""
    unit_ocr_conf: float = 0.0
    is_noise: bool = False
    noise_reason: str | None = None
    scope_idx: int | None = None


@dataclass
class Scope:
    idx: int
    bbox: list[int]
    confidence: float


@dataclass
class TextGroup:
    idx: int
    unit_indices: list[int]
    scoped: bool
    scope_idx: int | None
    raw_bbox: list[int]
    ocr_bbox: list[int]
    fit_bbox: list[int]
    ocr_text: str = ""
    ocr_conf: float = 0.0
    accepted: bool = False
    reject_reason: str | None = None
    scope_bbox: list[int] | None = None


@dataclass
class PageScanState:
    image: np.ndarray
    width: int
    height: int
    units: list[TextUnit] = field(default_factory=list)
    scopes: list[Scope] = field(default_factory=list)
    groups: list[TextGroup] = field(default_factory=list)


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
    h = image.shape[0]
    units: list[TextRegion] = []
    for tile_y, tile_h in compute_tiles(h, PPOCR_MAX_TILE_HEIGHT):
        tile = image[tile_y:tile_y + tile_h]
        out = scanner._det.detect(tile)  # type: ignore[attr-defined]
        if tile_y:
            offset_regions(out.regions, tile_y, image)
        units.extend(out.regions)
    return deduplicate_regions(units)


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


def _ocr_crop_box(group_box: list[int], group_indices: set[int], all_boxes: list[list[int]], page_w: int, page_h: int) -> list[int]:
    x1, y1, x2, y2 = group_box
    pad = int(max(x2 - x1, y2 - y1) * 0.10)
    left = max(0, x1 - pad)
    top = max(0, y1 - pad)
    right = min(page_w, x2 + pad)
    bottom = min(page_h, y2 + pad)

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


def build_groups(state: PageScanState) -> None:
    active = [u.idx for u in state.units if not u.is_noise]
    boxes = [u.bbox for u in state.units]
    by_scope: dict[int, list[int]] = defaultdict(list)
    free: list[int] = []
    for i in active:
        scope_idx = state.units[i].scope_idx
        if scope_idx is None:
            free.append(i)
        else:
            by_scope[scope_idx].append(i)

    raw_groups: list[tuple[list[int], bool, int | None]] = []
    for scope_idx, indices in by_scope.items():
        for g in subgroup_text_blocks(indices, boxes, state.scopes[scope_idx].bbox):
            raw_groups.append((g, True, scope_idx))
    for g in subgroup_text_blocks(free, boxes, None):
        raw_groups.append((g, False, None))

    state.groups = []
    for gi, (indices, scoped, scope_idx) in enumerate(raw_groups):
        group_boxes = [boxes[i] for i in indices]
        raw = union_boxes(group_boxes)
        ocr = _ocr_crop_box(raw, set(indices), boxes, state.width, state.height)
        fit = expand(raw, fit_padding(group_boxes, state.width, state.height), state.width, state.height)
        scope_bbox = state.scopes[scope_idx].bbox if scope_idx is not None else None
        state.groups.append(TextGroup(gi, indices, scoped, scope_idx, raw, ocr, fit, scope_bbox=scope_bbox))


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
        if len(group.unit_indices) == 1 and group.ocr_conf < 0.35:
            return True, "free_singleton_low_conf"
        if area_ratio > 0.025 or width_ratio > 0.24 or height_ratio > 0.16:
            return True, "free_large_sfx_like"
        if len(text) <= 2 and group.ocr_conf < 0.80:
            return True, "free_short_low_conf"
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
