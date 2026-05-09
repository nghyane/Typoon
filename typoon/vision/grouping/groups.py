"""Group building, OCR, filtering, erase-mask generation, and export."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Protocol

import cv2
import numpy as np

from typoon.vision.types import (
    DetectedGroup,
    GroupState,
    ScanState,
    TextMask,
)

from .geometry import (
    box_iou,
    box_to_poly,
    containment_ratio,
    fit_to_scope,
    fit_padding,
    pad_box,
    union_boxes,
    x_gap, x_overlap,
    y_gap, y_overlap,
)

class _Scanner(Protocol):
    _det: object
    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


# ── Group building ────────────────────────────────────────────────


def subgroup_blocks(
    indices: list[int],
    boxes: list[list[int]],
    container: list[int] | None = None,
) -> list[list[int]]:
    """Split indices into spatially coherent sub-groups."""
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

    mode = "strict" if container is None else "normal"
    if container is not None:
        cw = max(1, container[2] - container[0])
        ch = max(1, container[3] - container[1])
        uw = max(1, text_union[2] - text_union[0])
        uh = max(1, text_union[3] - text_union[1])
        n = len(indices)
        all_inside = all(
            container[0] <= (boxes[i][0] + boxes[i][2]) / 2 <= container[2]
            and container[1] <= (boxes[i][1] + boxes[i][3]) / 2 <= container[3]
            for i in indices
        )
        # Don't short-circuit when there is a large vertical gap between
        # subsequent units inside the same scope: a peanut/eared bubble may
        # contain a small "ear" cluster (e.g. "OF COURSE!") plus the main
        # body, both fully inside the YOLO box, but separated by white
        # space that should split them into distinct groups.
        if all_inside and large_gaps == 0:
            return [list(indices)]
        if n <= 6 and uh / ch < 0.85 and uw / cw < 0.98 and large_gaps == 0:
            return [list(indices)]
        if n >= 5 and (uh / ch > 0.80 or large_gaps >= 2):
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
                sc = x_overlap(a, b) > 0.70 and y_gap(a, b) <= min_h * 0.45 and hr < 1.7
                sr = y_overlap(a, b) > 0.70 and x_gap(a, b) <= min_h * 1.10 and hr < 1.7
                ov = False
            else:
                sc = x_overlap(a, b) > 0.55 and y_gap(a, b) <= min_h * 0.85 and hr < 2.1
                sr = y_overlap(a, b) > 0.60 and x_gap(a, b) <= min_h * 1.50 and hr < 2.1
                ov = box_iou(a, b) > 0.12 or containment_ratio(a, b) > 0.35
            if sc or sr or ov:
                join(i, j)

    groups_map: dict[int, list[int]] = {}
    for i in indices:
        groups_map.setdefault(find(i), []).append(i)
    return [sorted(g, key=lambda i: (boxes[i][1], boxes[i][0])) for g in groups_map.values()]


def build_groups(state: ScanState) -> None:
    """Build TextGroup list from active units and scopes."""
    active = [u.idx for u in state.units if not u.is_noise]
    boxes = [u.bbox for u in state.units]
    angles = [_poly_angle(state.units[i].region.polygon) for i in range(len(state.units))]
    by_scope: dict[int, list[int]] = defaultdict(list)
    free: list[int] = []
    for i in active:
        (free if state.units[i].scope_idx is None else by_scope[state.units[i].scope_idx]).append(i)

    raw: list[tuple[list[int], bool, int | None]] = []

    for scope_idx, indices in by_scope.items():
        clusters = _cluster_by_angle(indices, angles)
        if len(clusters) == 1:
            for g in subgroup_blocks(indices, boxes, state.scopes[scope_idx].bbox):
                raw.append((g, True, scope_idx))
        else:
            largest = max(clusters, key=len)
            for c in clusters:
                if c is largest:
                    for g in subgroup_blocks(c, boxes, state.scopes[scope_idx].bbox):
                        raw.append((g, True, scope_idx))
                else:
                    free.extend(c)

    for c in _cluster_by_angle(free, angles):
        for g in subgroup_blocks(c, boxes, None):
            raw.append((g, False, None))

    state.groups = []
    for gi, (indices, scoped, scope_idx) in enumerate(raw):
        group_boxes = [boxes[i] for i in indices]
        raw_bbox = union_boxes(group_boxes)
        scope_bbox = state.scopes[scope_idx].bbox if scope_idx is not None else None
        ocr_box = _ocr_crop_box(raw_bbox, set(indices), boxes, state.width, state.height, scope_bbox)
        pad = fit_padding(group_boxes, state.width, state.height)
        fit = pad_box(raw_bbox, pad, state.width, state.height)
        if scope_bbox is not None:
            fit = [max(fit[0], scope_bbox[0]), max(fit[1], scope_bbox[1]),
                   min(fit[2], scope_bbox[2]), min(fit[3], scope_bbox[3])]
            fit = fit_to_scope(fit, scope_bbox, pad)
        med_angle = float(np.median([angles[i] for i in indices])) if indices else 0.0
        det_conf = max(
            (state.units[i].region.confidence for i in indices), default=0.0,
        )
        state.groups.append(GroupState(gi, indices, scoped, scope_idx, raw_bbox, ocr_box, fit,
                                       scope_bbox=scope_bbox, median_angle=med_angle,
                                       det_conf=float(det_conf)))


def ocr_groups(state: ScanState, scanner: _Scanner) -> None:
    """Run full OCR on each group crop, with rotation retry for angled text.

    Pre-OCR noise filter on unscoped groups (no YOLO bubble): pure geometry
    short-circuit so we don't pay OCR cost on detect-noise / page artifacts.
    Scoped groups always go through — their geometry comes from the bubble
    detector and is trusted. The kept-set still passes through `_should_skip`
    after OCR for text-quality decisions.
    """
    raw = bool(getattr(scanner, "wants_raw", False))
    # Pass 1: OCR all crops at detected angle
    crops, groups = [], []
    for g in state.groups:
        x1, y1, x2, y2 = g.ocr_bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 5 or bh < 5:
            continue
        if not g.scoped and _is_geometric_noise(bw, bh, g.median_angle):
            g.text = ""
            g.confidence = 0.0
            continue
        if not g.scoped and _is_stripe_cluster(state.units, list(g.unit_indices)):
            g.text = ""
            g.confidence = 0.0
            continue
        crops.append(_rotate_crop(state.image[y1:y2, x1:x2], g.median_angle, raw=raw))
        groups.append(g)
    for g, (text, conf) in zip(groups, scanner._ocr_crops(crops) if crops else []):
        g.text = (text or "").strip()
        g.confidence = float(conf)

    # Pass 2: retry low-confidence angled groups with candidate angle offsets
    retry_groups, retry_crops = [], []
    for g in groups:
        if g.confidence >= 0.8 or abs(g.median_angle) < 3:
            continue
        x1, y1, x2, y2 = g.ocr_bbox
        base = state.image[y1:y2, x1:x2]
        for offset in (-12, -8, -4, 4, 8, 12):
            retry_groups.append(g)
            retry_crops.append(_rotate_crop(base, g.median_angle + offset, raw=raw))

    if retry_crops:
        for g, (text, conf) in zip(retry_groups, scanner._ocr_crops(retry_crops)):
            text = (text or "").strip()
            if conf > g.confidence:
                g.text = text
                g.confidence = float(conf)


def _is_geometric_noise(bw: int, bh: int, angle: float) -> bool:
    """Pre-OCR filter on bbox geometry alone — unscoped groups only.

    Mirrors the geometry checks in `_should_skip` (`free_tiny`,
    `free_skewed`) so they run before OCR instead of after. Verified on
    fixture chapters: scoped dialogue has min(W,H) ≥ 113 and angle ≈ 0°,
    while unscoped detect-noise spans the full filter range.
    """
    if min(bw, bh) < 20:
        return True
    if abs(angle) > 20.0:
        return True
    return False


def _polygon_axis(poly: list[list[float]]) -> tuple[float, float]:
    """Return (long_axis_angle_deg, aspect) of polygon's rotated rect.

    `aspect` = long_side / short_side (≥ 1.0).
    `long_axis_angle_deg` is the long edge's angle relative to horizontal,
    in (-90°, 90°].
    """
    if len(poly) < 4:
        return 0.0, 1.0
    tl, tr, br, bl = poly[0], poly[1], poly[2], poly[3]
    top_w = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    side_h = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    if top_w >= side_h:
        ang = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))
        long_, short_ = top_w, max(1.0, side_h)
    else:
        ang = math.degrees(math.atan2(bl[1] - tl[1], bl[0] - tl[0]))
        long_, short_ = side_h, max(1.0, top_w)
    return ang, long_ / short_


def _axis_deviation(angle_deg: float) -> float:
    """Distance from nearest cardinal axis (0° or 90°), in degrees [0, 45]."""
    a = abs(angle_deg) % 180.0
    return min(a, abs(a - 90.0), 180.0 - a)


def _is_stripe_polygon(poly: list[list[float]]) -> bool:
    """Detector polygon shape that matches a hatching/screentone stroke.

    PP-OCR det fires on dense parallel ink strokes (screentone, hatching,
    motion lines) and returns thin diagonal stripes. Real glyph clusters
    after unclip+merge are roughly axis-aligned — vertical Japanese
    columns sit at 90° (axis_dev ≈ 0), horizontal Latin runs at 0°.
    A polygon that is BOTH narrow (aspect ≥ 3.5) AND tilted off both
    cardinal axes (≥ 12°) is structurally a stroke fragment, not text.
    """
    ang, aspect = _polygon_axis(poly)
    return aspect >= 3.5 and _axis_deviation(ang) >= 12.0


def _is_stripe_cluster(units, indices: list[int]) -> bool:
    """Group looks like a hatching cluster: every unit polygon is a stripe.

    All-or-nothing on purpose — one real glyph fragment in the group means
    the cluster carries information. This avoids killing a real bubble
    that happens to neighbor a stripe.
    """
    if not indices:
        return False
    return all(_is_stripe_polygon(units[i].region.polygon) for i in indices)



def _rotate_crop(crop: np.ndarray, angle: float, *, raw: bool = False) -> np.ndarray:
    """Rotate crop to straighten text, then binarize for cleaner OCR.

    `raw=True` skips binarization for backends (e.g. manga-ocr) trained
    on natural grayscale manga, where adaptive-threshold preprocessing
    degrades accuracy.
    """
    if abs(angle) >= 3:
        ch, cw = crop.shape[:2]
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), -angle, 1.0)
        crop = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    if raw:
        return crop
    # Adaptive binarization removes bubble-edge noise and improves manga font OCR.
    # Detect polarity: dark-on-light (manga/manhwa white bubble) vs light-on-dark
    # (dark/colored bubble background). THRESH_BINARY_INV for the latter so text
    # pixels always become black in the binarized crop fed to OCR.
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    thresh_type = (
        cv2.THRESH_BINARY if gray.mean() > 127 else cv2.THRESH_BINARY_INV
    )
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 11, 2)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)


def filter_groups(state: ScanState) -> None:
    """Accept or reject each group based on OCR quality and geometry."""
    for g in state.groups:
        skip, reason = _should_skip(g, state.width, state.height)
        g.accepted = not skip
        g.reject_reason = reason


def export_groups(state: ScanState) -> list[DetectedGroup]:
    """Export accepted groups as DetectedGroup (pipeline output type)."""
    out: list[DetectedGroup] = []
    for g in state.groups:
        if not g.accepted:
            continue
        text_masks = [
            state.units[i].region.mask
            for i in g.unit_indices
            if state.units[i].region.mask is not None
        ]
        mask_box = _masks_bbox(text_masks, g.fit_bbox)
        render_poly = box_to_poly(g.fit_bbox)
        erase_masks = build_erase_masks(
            text_masks, mode="glow" if _is_uppercase_heavy(g.text) else "normal"
        )
        erase_box = _masks_bbox(erase_masks, g.fit_bbox)
        out.append(DetectedGroup(
            text=g.text,
            confidence=g.confidence,
            text_polygon=box_to_poly(g.raw_bbox),
            render_polygon=render_poly,
            text_box=g.raw_bbox,
            mask_box=mask_box,
            fit_box=g.fit_bbox,
            erase_box=erase_box,
            scope_box=g.scope_bbox,
            scope_confidence=(
                state.scopes[g.scope_idx].confidence if g.scope_idx is not None else None
            ),
            text_masks=text_masks,
            erase_masks=erase_masks,
            source="scoped" if g.scoped else "free",
            unit_indices=list(g.unit_indices),
            accepted=g.accepted,
            reject_reason=g.reject_reason,
            shape_kind=g.shape_kind,
        ))
    return out


# ── Erase masks ───────────────────────────────────────────────────


def build_erase_masks(masks: list[TextMask], *, mode: str = "normal") -> list[TextMask]:
    out = []
    for mask in masks:
        h = mask.image.shape[0]
        pad = int(max(5, min(h * 0.16, 20))) if mode == "glow" else int(max(3, min(h * 0.10, 14)))
        out.append(_dilate_mask(mask, pad))
    return out


def _dilate_mask(mask: TextMask, pad: int) -> TextMask:
    if pad <= 0:
        return mask
    mh, mw = mask.image.shape[:2]
    expanded = np.zeros((mh + pad * 2, mw + pad * 2), dtype=np.uint8)
    expanded[pad:pad + mh, pad:pad + mw] = mask.image
    ksize = pad * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return TextMask(x=mask.x - pad, y=mask.y - pad,
                    image=cv2.dilate(expanded, kernel, iterations=1))


# ── Internal helpers ──────────────────────────────────────────────


def _poly_angle(polygon: list[list[float]]) -> float:
    if len(polygon) < 4:
        return 0.0
    bl, br = polygon[3], polygon[2]
    return math.degrees(math.atan2(br[1] - bl[1], br[0] - bl[0]))


def _angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def _cluster_by_angle(indices: list[int], angles: list[float], threshold: float = 20.0) -> list[list[int]]:
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    reps: list[float] = [angles[indices[0]]]
    for i in indices[1:]:
        ang = angles[i]
        placed = False
        for ci, ca in enumerate(reps):
            if _angle_diff(ang, ca) <= threshold:
                clusters[ci].append(i)
                reps[ci] = (ca * (len(clusters[ci]) - 1) + ang) / len(clusters[ci])
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(ang)
    return clusters


def _ocr_crop_box(
    group_box: list[int], group_indices: set[int], all_boxes: list[list[int]],
    page_w: int, page_h: int, scope_bbox: list[int] | None = None,
) -> list[int]:
    x1, y1, x2, y2 = group_box

    # No OCR padding: the raw_bbox already contains all detected ink pixels.
    # Adding margin risks pulling in bubble outline ink (for unscoped groups)
    # or adjacent panel content. Noise fragments like 'ic WHERE' are a
    # grouping artefact, not a tight-crop issue — fix them in grouping, not here.
    l, t = x1, y1
    r, b = x2, y2

    # Clip to scope boundary (hard wall — do not read outside the bubble)
    if scope_bbox is not None:
        sx1, sy1, sx2, sy2 = scope_bbox
        l = max(l, sx1)
        t = max(t, sy1)
        r = min(r, sx2)
        b = min(b, sy2)

    # Clip to page boundary
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, page_w)
    b = min(b, page_h)

    # Clip away neighboring text regions so their pixels don't contaminate
    # the crop. Applied after padding so the pad does not reach into neighbors.
    for i, other in enumerate(all_boxes):
        if i in group_indices:
            continue
        ox1, oy1, ox2, oy2 = other
        if min(b, oy2) > max(t, oy1):
            if ox2 <= x1 and ox2 > l: l = ox2 + 1
            elif ox1 >= x2 and ox1 < r: r = ox1 - 1
        if min(r, ox2) > max(l, ox1):
            if oy2 <= y1 and oy2 > t: t = oy2 + 1
            elif oy1 >= y2 and oy1 < b: b = oy1 - 1

    return [l, t, r, b]


def _masks_bbox(masks: list[TextMask], fallback: list[int]) -> list[int]:
    boxes = []
    for m in masks:
        mh, mw = m.image.shape[:2]
        boxes.append([int(m.x), int(m.y), int(m.x + mw), int(m.y + mh)])
    return union_boxes(boxes) if boxes else fallback


def _is_uppercase_heavy(text: str) -> bool:
    alpha = sum(ch.isalpha() for ch in text)
    return alpha >= 12 and sum(ch.isupper() for ch in text) / max(1, alpha) >= 0.55


def _required_proof(short_side: int) -> float:
    """Linear ramp: tiny fragments (~20px) need 0.65, large (~100px+) need 0.85.

    Encodes the principle that bigger claims need stronger proof. A real
    glyph cluster at 100+px should light up both detectors; an artwork
    fragment that happens to look kana-shaped (kẹp tóc hình "3" → "と")
    sits in the mid-confidence valley both detectors fall into when given
    a non-text input.
    """
    s = max(20, min(100, short_side))
    return 0.65 + (s - 20) / 80.0 * (0.85 - 0.65)


def _should_skip(g: GroupState, pw: int, ph: int) -> tuple[bool, str | None]:
    """Group-level accept/reject.

    Two responsibilities only:
      1. text quality   — empty, no-alnum (cheap textual checks)
      2. is-text proof  — for unscoped groups (no YOLO bubble), require both
         detectors to agree, scaled by claim size.

    Removed the older `free_low_conf`, `free_short_low_conf`,
    `free_large_sfx_like`, `free_angled_low_conf`, `_looks_like_narration`,
    `_looks_like_system_card` heuristics. Those layered ocr_conf thresholds
    by ad-hoc text shape — but `manga-ocr`/PP-OCR-rec are generative and
    have no "no-text" class, so ocr_conf alone never reliably answers
    "is this text". `det_conf` is the trained is-text signal; combining
    both via geometric mean and ramping by size replaces all of them.
    """
    text = g.text.strip()
    if not text:
        return True, "ocr_empty"
    if not sum(ch.isalnum() for ch in text):
        return True, "ocr_no_alnum"
    bw = max(1, g.raw_bbox[2] - g.raw_bbox[0])
    bh = max(1, g.raw_bbox[3] - g.raw_bbox[1])
    if not g.scoped:
        if abs(g.median_angle) > 20.0:
            return True, "free_skewed"
        if bw < 20 or bh < 20:
            return True, "free_tiny"
        proof = math.sqrt(max(0.0, g.det_conf) * max(0.0, g.confidence))
        if proof < _required_proof(min(bw, bh)):
            return True, "free_weak_proof"
    return False, None


