"""Unit detection, OCR filtering, and scope-based splitting."""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from typoon.vision.tiling import compute_tiles_2d, deduplicate_regions, offset_regions_2d
from typoon.vision.types import PageScanState, Scope, TextMask, TextRegion, TextUnit

from .geometry import bbox

PPOCR_MAX_TILE_HEIGHT = 2048
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_SPLIT_MIN_COVERAGE = 0.25


class _Scanner(Protocol):
    _det: object
    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


def detect_raw_text_units(scanner: _Scanner, image: np.ndarray) -> list[TextRegion]:
    h, w = image.shape[:2]
    all_regions: list[TextRegion] = []
    for tx, ty, tw, th in compute_tiles_2d(h, w):
        tile = image[ty:ty + th, tx:tx + tw]
        out = scanner._det.detect(tile)  # type: ignore[attr-defined]
        if tx or ty:
            offset_regions_2d(out.regions, tx, ty, image)
        all_regions.extend(out.regions)
    return deduplicate_regions(all_regions)


def unit_quality(
    unit: TextRegion,
    box: list[int],
    ocr_text: str,
    ocr_conf: float,
    *,
    filter_cjk: bool = False,
) -> tuple[bool, str | None]:
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


def detect_units(state: PageScanState, scanner: _Scanner) -> None:
    regions = detect_raw_text_units(scanner, state.image)
    state.units = [TextUnit(idx=i, region=r, bbox=bbox(r.polygon)) for i, r in enumerate(regions)]


def ocr_units_for_filtering(state: PageScanState, scanner: _Scanner) -> None:
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


def detect_scopes(state: PageScanState, yolo_model, yolo_imgsz: int, yolo_conf: float) -> None:
    if yolo_model is None:
        return
    from typoon.vision.bubble_scope import detect_bubble_scopes
    scopes = detect_bubble_scopes(yolo_model, state.image, imgsz=yolo_imgsz, conf=yolo_conf)
    state.scopes = [Scope(idx=i, bbox=s.bbox, confidence=s.confidence) for i, s in enumerate(scopes)]


def assign_units_to_scopes(state: PageScanState) -> None:
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


def split_units_crossing_scopes(state: PageScanState) -> None:
    """Split horizontal units spanning 2+ YOLO scope boundaries."""
    if not state.scopes or not state.units:
        return

    widths = [max(1, u.bbox[2] - u.bbox[0]) for u in state.units]
    median_w = float(np.median(widths)) if widths else 1.0

    new_units: list[TextUnit] = []
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
            sx1, _, sx2, _ = s.bbox
            ox1, ox2 = max(ux1, sx1), min(ux2, sx2)
            if ox2 <= ox1:
                continue
            if (ox2 - ox1) / uw >= _SPLIT_MIN_COVERAGE:
                qualifying.append((si, ox1, ox2))

        if len(qualifying) >= 2:
            filtered = []
            for qi in range(len(qualifying)):
                si_x1, _, si_x2, _ = state.scopes[qualifying[qi][0]].bbox
                dominated = any(
                    state.scopes[qualifying[qj][0]].bbox[0] <= si_x1
                    and si_x2 <= state.scopes[qualifying[qj][0]].bbox[2]
                    for qj in range(len(qualifying)) if qi != qj
                )
                if not dominated:
                    filtered.append(qualifying[qi])
            qualifying = filtered

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
                mh, mw = m.image.shape[:2]
                mc1 = max(0, sub_x1 - m.x)
                mc2 = min(mw, sub_x2 - m.x)
                if mc2 > mc1:
                    sub_mask = TextMask(x=m.x + mc1, y=m.y, image=m.image[:, mc1:mc2].copy())

            sub_region = TextRegion(
                polygon=sub_poly, crop=sub_crop,
                confidence=u.region.confidence, mask=sub_mask,
            )
            new_units.append(TextUnit(
                idx=idx, region=sub_region, bbox=[sub_x1, uy1, sub_x2, uy2],
                unit_ocr_text=u.unit_ocr_text, unit_ocr_conf=u.unit_ocr_conf,
                is_noise=u.is_noise, noise_reason=u.noise_reason,
            ))
            idx += 1
            prev_x2 = ox2

    state.units = new_units
