"""Unit detection, OCR filtering, and scope-based splitting."""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from typoon.vision.tiling import compute_tiles_2d, deduplicate_regions, offset_regions_2d
from typoon.vision.types import ScanState, ScopeState, TextMask, TextRegion, UnitState

from .geometry import poly_bbox

CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_SPLIT_MIN_COVERAGE = 0.25


class _Scanner(Protocol):
    _det: object
    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


def detect_units(state: ScanState, scanner: _Scanner) -> None:
    """Detect text regions and populate state.units."""
    h, w = state.image.shape[:2]
    all_regions: list[TextRegion] = []
    for tx, ty, tw, th in compute_tiles_2d(h, w):
        tile = state.image[ty:ty + th, tx:tx + tw]
        out = scanner._det.detect(tile)  # type: ignore[attr-defined]
        if tx or ty:
            offset_regions_2d(out.regions, tx, ty, state.image)
        all_regions.extend(out.regions)
    regions = deduplicate_regions(all_regions)
    state.units = [UnitState(idx=i, region=r, bbox=poly_bbox(r.polygon)) for i, r in enumerate(regions)]


def ocr_units(state: ScanState, scanner: _Scanner) -> None:
    """Run quick OCR pass on unit crops for noise filtering."""
    crops = [u.region.crop for u in state.units]
    for u, (text, conf) in zip(state.units, scanner._ocr_crops(crops) if crops else []):
        u.text = (text or "").strip()
        u.confidence = float(conf)


def filter_units(state: ScanState) -> None:
    """Mark noisy units — too small, low confidence, non-text chars."""
    for u in state.units:
        ok, reason = _unit_quality(u.region, u.bbox, u.text, u.confidence)
        u.is_noise = not ok
        u.noise_reason = reason


def detect_scopes(state: ScanState, yolo_model, imgsz: int, conf: float) -> None:
    """Run YOLO bubble scope detection and populate state.scopes."""
    if yolo_model is None:
        return
    from typoon.vision.bubble_scope import detect_bubble_scopes
    scopes = detect_bubble_scopes(yolo_model, state.image, imgsz=imgsz, conf=conf)
    state.scopes = [ScopeState(idx=i, bbox=s.bbox, confidence=s.confidence) for i, s in enumerate(scopes)]


def assign_scopes(state: ScanState) -> None:
    """Assign each unit to its best YOLO scope."""
    if not state.scopes:
        return
    from typoon.vision.bubble_scope import BubbleScope
    from typoon.vision.bubble_scope import assign_units_to_scopes as _assign
    assignments = _assign(
        [u.bbox for u in state.units],
        [BubbleScope(bbox=s.bbox, confidence=s.confidence) for s in state.scopes],
    )
    for u, scope_idx in zip(state.units, assignments):
        u.scope_idx = scope_idx


def split_units(state: ScanState) -> None:
    """Split wide units spanning 2+ YOLO scope boundaries."""
    if not state.scopes or not state.units:
        return

    widths = [max(1, u.bbox[2] - u.bbox[0]) for u in state.units]
    median_w = float(np.median(widths)) if widths else 1.0

    new_units: list[UnitState] = []
    idx = 0
    for u in state.units:
        ux1, uy1, ux2, uy2 = u.bbox
        uw = max(1, ux2 - ux1)

        if uw < median_w * 1.5:
            u.idx = idx
            new_units.append(u)
            idx += 1
            continue

        qualifying: list[tuple[int, int, int]] = []
        for si, s in enumerate(state.scopes):
            sx1, sy1, sx2, sy2 = s.bbox
            # scope must y-overlap with unit — prevents scopes from adjacent
            # bubbles (different y-band) from triggering a spurious split
            if sy2 <= uy1 or sy1 >= uy2:
                continue
            ox1, ox2 = max(ux1, sx1), min(ux2, sx2)
            if ox2 > ox1 and (ox2 - ox1) / uw >= _SPLIT_MIN_COVERAGE:
                qualifying.append((si, ox1, ox2))

        if len(qualifying) >= 2:
            qualifying = [
                q for qi, q in enumerate(qualifying)
                if not any(
                    state.scopes[qualifying[qj][0]].bbox[0] <= state.scopes[q[0]].bbox[0]
                    and state.scopes[q[0]].bbox[2] <= state.scopes[qualifying[qj][0]].bbox[2]
                    for qj in range(len(qualifying)) if qi != qj
                )
            ]

        if len(qualifying) < 2:
            u.idx = idx
            new_units.append(u)
            idx += 1
            continue

        qualifying.sort(key=lambda t: t[1])
        crop = u.region.crop

        prev_x2 = None
        for qi, (si, ox1, ox2) in enumerate(qualifying):
            sub_x1 = ux1 if prev_x2 is None else (prev_x2 + ox1) // 2
            sub_x2 = min(ox2 if qi == len(qualifying) - 1 else (ox2 + qualifying[qi + 1][1]) // 2, ux2)
            if sub_x2 - sub_x1 < 10:
                prev_x2 = ox2
                continue

            c1 = max(0, sub_x1 - ux1)
            c2 = min(crop.shape[1], sub_x2 - ux1)
            sub_crop = crop[:, c1:c2] if c2 > c1 else crop

            sub_poly = [
                [min(max(float(p[0]), float(sub_x1)), float(sub_x2)), float(p[1])]
                for p in u.region.polygon
            ]

            sub_mask = None
            if u.region.mask is not None:
                m = u.region.mask
                mw = m.image.shape[1]
                mc1, mc2 = max(0, sub_x1 - m.x), min(mw, sub_x2 - m.x)
                if mc2 > mc1:
                    sub_mask = TextMask(x=m.x + mc1, y=m.y, image=m.image[:, mc1:mc2].copy())

            new_units.append(UnitState(
                idx=idx,
                region=TextRegion(polygon=sub_poly, crop=sub_crop,
                                  confidence=u.region.confidence, mask=sub_mask),
                bbox=[sub_x1, uy1, sub_x2, uy2],
                text=u.text, confidence=u.confidence,
                is_noise=u.is_noise, noise_reason=u.noise_reason,
            ))
            idx += 1
            prev_x2 = ox2

    state.units = new_units


def _unit_quality(
    unit: TextRegion, box: list[int], text: str, conf: float,
    *, filter_cjk: bool = False,
) -> tuple[bool, str | None]:
    w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    if w < 4 or h < 4 or w * h < 24:
        return False, "tiny"
    if unit.confidence < 0.20 and (unit.mask is None or not np.count_nonzero(unit.mask.image)):
        return False, "low_det_low_mask"
    t = (text or "").strip()
    alnum = sum(ch.isalnum() for ch in t)
    if filter_cjk and CJK_RE.search(t):
        return False, "cjk_filtered"
    if not t and conf < 0.15 and unit.confidence < 0.45:
        return False, "ocr_empty_low_conf"
    if t and alnum == 0 and conf < 0.35:
        return False, "ocr_non_text_chars"
    return True, None


