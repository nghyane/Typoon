"""PP-OCR + YOLO union-find grouper.

Combines DBNet text fragments (TextBlock with polygon, no text) with YOLO
bubble-scope hints to merge fragments into coherent bubbles. The merge
logic is the carry-over from the legacy grouping/units.py + groups.py
modules, refactored to operate on the unified TextBlock contract.

Used by the `offline` and `manga_ja` presets when Lens is unavailable or
inappropriate (e.g. no internet, language-specific OCR like manga-ocr).
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..contracts import BubbleGroup, DetectionResult, TextBlock, TextMask


__all__ = ["PPOCRYoloUnionFindGrouper"]


# ─── Tunables ─────────────────────────────────────────────────────────────


_BUBBLE_SCOPE_IMGSZ = 640
_BUBBLE_SCOPE_CONF = 0.3
_ERASE_DILATE_FRACTION_NORMAL = 0.10
_ERASE_DILATE_FRACTION_GLOW   = 0.16
_ERASE_DILATE_MIN_PX = 3
_ERASE_DILATE_MAX_PX = 20


# ─── Internal mutable state during grouping ───────────────────────────────


@dataclass(slots=True)
class _Unit:
    idx:        int
    bbox:       tuple[int, int, int, int]
    polygon:    tuple[tuple[float, float], ...]
    confidence: float
    angle_deg:  float
    scope_idx:  int | None = None


@dataclass(slots=True)
class _Scope:
    idx:        int
    bbox:       tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class _GroupBuild:
    indices:     list[int]
    scope_idx:   int | None
    bbox:        tuple[int, int, int, int]
    text_masks:  tuple[TextMask, ...] = ()


# ─── Grouper ──────────────────────────────────────────────────────────────


class PPOCRYoloUnionFindGrouper:
    """Legacy union-find merge of PP-OCR fragments under YOLO bubble scopes.

    Lazy-loads YOLO model on first call. Sync internals run inside
    asyncio.to_thread; outer interface is async.
    """

    name = "ppocr_yolo_union_find"

    def __init__(self, *, models_dir: Path) -> None:
        self._models_dir = Path(models_dir)
        self._yolo = None

    async def group(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        lang: str | None,
    ) -> tuple[BubbleGroup, ...]:
        return await asyncio.to_thread(self._build, image, detection)

    # Internal sync pipeline

    def _build(self, image: np.ndarray, detection: DetectionResult) -> tuple[BubbleGroup, ...]:
        units = [_block_to_unit(i, b) for i, b in enumerate(detection.blocks)]
        if not units:
            return ()
        scopes = self._detect_scopes(image)
        for u in units:
            u.scope_idx = _assign_scope(u.bbox, scopes)
        groups = _merge_into_groups(units, scopes)
        return tuple(_group_to_bubble(g, units) for g in groups)

    def _detect_scopes(self, image: np.ndarray) -> list[_Scope]:
        model = self._get_yolo()
        if model is None:
            return []
        results = model.predict(
            image, imgsz=_BUBBLE_SCOPE_IMGSZ, conf=_BUBBLE_SCOPE_CONF,
            iou=0.5, verbose=False,
        )
        scopes: list[_Scope] = []
        for r in results:
            if r.boxes is None:
                continue
            for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                scopes.append(_Scope(
                    idx=len(scopes),
                    bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                    confidence=float(c),
                ))
        return _merge_scopes(scopes)

    def _get_yolo(self):
        if self._yolo is not None:
            return self._yolo
        try:
            from ultralytics import YOLO
        except ImportError:
            return None
        path = (
            self._models_dir / "bubble-scope-yolov8m.mlpackage"
            if sys.platform == "darwin"
            else self._models_dir / "bubble-scope-yolov8m.pt"
        )
        if not path.exists():
            return None
        self._yolo = YOLO(str(path))
        return self._yolo


# ─── Block → unit ─────────────────────────────────────────────────────────


def _block_to_unit(idx: int, block: TextBlock) -> _Unit:
    polygon = block.polygon or _bbox_polygon(block.bbox)
    return _Unit(
        idx=idx,
        bbox=block.bbox,
        polygon=polygon,
        confidence=block.confidence,
        angle_deg=_polygon_angle(polygon),
    )


def _bbox_polygon(bbox: tuple[int, int, int, int]) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = bbox
    return (
        (float(x1), float(y1)),
        (float(x2), float(y1)),
        (float(x2), float(y2)),
        (float(x1), float(y2)),
    )


def _polygon_angle(poly: tuple[tuple[float, float], ...]) -> float:
    if len(poly) < 4:
        return 0.0
    bl, br = poly[3], poly[2]
    return math.degrees(math.atan2(br[1] - bl[1], br[0] - bl[0]))


# ─── Scope assignment ─────────────────────────────────────────────────────


def _merge_scopes(scopes: list[_Scope]) -> list[_Scope]:
    """Drop overlapping YOLO bbox duplicates (IoU > 0.5)."""
    if len(scopes) <= 1:
        return scopes
    sorted_s = sorted(scopes, key=lambda s: -s.confidence)
    kept: list[_Scope] = []
    for s in sorted_s:
        if not any(_iou(s.bbox, k.bbox) > 0.5 for k in kept):
            kept.append(_Scope(idx=len(kept), bbox=s.bbox, confidence=s.confidence))
    return kept


def _assign_scope(bbox: tuple[int, int, int, int], scopes: list[_Scope]) -> int | None:
    if not scopes:
        return None
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for s in scopes:
        if s.bbox[0] <= cx <= s.bbox[2] and s.bbox[1] <= cy <= s.bbox[3]:
            return s.idx
    return None


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, aa + ab - inter)


# ─── Union-find merge ─────────────────────────────────────────────────────


def _merge_into_groups(units: list[_Unit], scopes: list[_Scope]) -> list[_GroupBuild]:
    by_scope: dict[int, list[int]] = defaultdict(list)
    free: list[int] = []
    for u in units:
        if u.scope_idx is None:
            free.append(u.idx)
        else:
            by_scope[u.scope_idx].append(u.idx)

    groups: list[_GroupBuild] = []
    for scope_idx, indices in by_scope.items():
        for cluster in _split_by_angle(indices, units):
            groups.append(_assemble_group(cluster, units, scope_idx))

    for cluster in _split_by_angle(free, units):
        for sub in _union_find(cluster, units):
            groups.append(_assemble_group(sub, units, None))

    return groups


def _split_by_angle(indices: list[int], units: list[_Unit], threshold: float = 20.0) -> list[list[int]]:
    if not indices:
        return []
    clusters: list[list[int]] = [[indices[0]]]
    reps: list[float] = [units[indices[0]].angle_deg]
    for i in indices[1:]:
        ang = units[i].angle_deg
        placed = False
        for ci, ca in enumerate(reps):
            diff = abs(ang - ca) % 180
            diff = min(diff, 180 - diff)
            if diff <= threshold:
                clusters[ci].append(i)
                reps[ci] = (ca * (len(clusters[ci]) - 1) + ang) / len(clusters[ci])
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(ang)
    return clusters


def _union_find(indices: list[int], units: list[_Unit]) -> list[list[int]]:
    """Free-cluster spatial union-find (no scope container)."""
    if not indices:
        return []
    parent = {i: i for i in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for ni, i in enumerate(indices):
        a = units[i].bbox
        ah = max(1, a[3] - a[1])
        for j in indices[ni + 1:]:
            b = units[j].bbox
            bh = max(1, b[3] - b[1])
            min_h = max(1, min(ah, bh))
            hr = max(ah, bh) / min_h
            sc = _x_overlap(a, b) > 0.55 and _y_gap(a, b) <= min_h * 0.85 and hr < 2.1
            sr = _y_overlap(a, b) > 0.60 and _x_gap(a, b) <= min_h * 1.50 and hr < 2.1
            if sc or sr:
                join(i, j)

    out: dict[int, list[int]] = defaultdict(list)
    for i in indices:
        out[find(i)].append(i)
    return list(out.values())


def _assemble_group(
    indices: list[int], units: list[_Unit], scope_idx: int | None,
) -> _GroupBuild:
    bboxes = [units[i].bbox for i in indices]
    bbox = (
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    )
    return _GroupBuild(indices=list(indices), scope_idx=scope_idx, bbox=bbox)


def _x_overlap(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def _y_overlap(a, b) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def _x_gap(a, b) -> float:
    return max(0, max(a[0], b[0]) - min(a[2], b[2]))


def _y_gap(a, b) -> float:
    return max(0, max(a[1], b[1]) - min(a[3], b[3]))


# ─── Group → BubbleGroup ──────────────────────────────────────────────────


def _group_to_bubble(g: _GroupBuild, units: list[_Unit]) -> BubbleGroup:
    text_masks = tuple(_polygon_to_mask(units[i].polygon) for i in g.indices)
    erase_masks = tuple(
        _dilate_mask(m, _erase_dilate_px(g.bbox))
        for m in text_masks
    )
    return BubbleGroup(
        bbox=g.bbox,
        polygon=_bbox_polygon(g.bbox),
        text="",  # filled by recognizer
        confidence=max((units[i].confidence for i in g.indices), default=0.0),
        text_masks=text_masks,
        erase_masks=erase_masks,
        source="ppocr_yolo",
    )


def _polygon_to_mask(polygon: tuple[tuple[float, float], ...]) -> TextMask:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[int(p[0]) - x1, int(p[1]) - y1] for p in polygon], dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    return TextMask(x=x1, y=y1, image=img)


def _erase_dilate_px(bbox: tuple[int, int, int, int]) -> int:
    short = max(1, min(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return int(max(_ERASE_DILATE_MIN_PX,
                   min(short * _ERASE_DILATE_FRACTION_NORMAL, _ERASE_DILATE_MAX_PX)))


def _dilate_mask(mask: TextMask, pad: int) -> TextMask:
    if pad <= 0:
        return mask
    mh, mw = mask.image.shape[:2]
    expanded = np.zeros((mh + pad * 2, mw + pad * 2), dtype=np.uint8)
    expanded[pad:pad + mh, pad:pad + mw] = mask.image
    ksize = pad * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(expanded, kernel, iterations=1)
    return TextMask(x=mask.x - pad, y=mask.y - pad, image=dilated)
