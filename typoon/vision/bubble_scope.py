"""YOLO bubble scope detection for text grouping.

Provides bubble bounding boxes as scope hints for PP-OCR text grouping.
YOLO is only a scope signal -- it does not define final FIT regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BubbleScope:
    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float


def _center_inside(box: list[int], outer: list[int]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _inside_ratio(inner: list[int], outer: list[int]) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def _box_iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, aa + bb - inter)


def detect_bubble_scopes(model, image: np.ndarray, *, imgsz: int = 640, conf: float = 0.3) -> list[BubbleScope]:
    """Run YOLO bubble detection, return bounding boxes only."""
    results = model.predict(image, imgsz=imgsz, conf=conf, iou=0.5, verbose=False)
    scopes: list[BubbleScope] = []
    for r in results:
        if r.boxes is None:
            continue
        for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            scopes.append(BubbleScope(
                bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                confidence=float(c),
            ))
    return _merge_overlapping(scopes)


def _merge_overlapping(scopes: list[BubbleScope]) -> list[BubbleScope]:
    """Merge duplicate/overlapping YOLO boxes."""
    if len(scopes) <= 1:
        return scopes
    parent = list(range(len(scopes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(scopes)):
        for j in range(i + 1, len(scopes)):
            bi, bj = scopes[i].bbox, scopes[j].bbox
            if _box_iou(bi, bj) >= 0.25 or _inside_ratio(bi, bj) >= 0.65 or _inside_ratio(bj, bi) >= 0.65:
                join(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(len(scopes)):
        groups.setdefault(find(i), []).append(i)

    merged: list[BubbleScope] = []
    for indices in groups.values():
        boxes = [scopes[i].bbox for i in indices]
        conf = max(scopes[i].confidence for i in indices)
        ub = [min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes)]
        merged.append(BubbleScope(bbox=ub, confidence=conf))
    return merged


def _x_overlap(a: list[int], b: list[int]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def _y_overlap(a: list[int], b: list[int]) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1, min(a[3] - a[1], b[3] - b[1]))


def _gap_y(a: list[int], b: list[int]) -> int:
    return max(0, max(a[1], b[1]) - min(a[3], b[3]))


def _gap_x(a: list[int], b: list[int]) -> int:
    return max(0, max(a[0], b[0]) - min(a[2], b[2]))


def _neighbor_support(text_boxes: list[list[int]], assignments: list[int | None], unit_idx: int, scope_idx: int) -> int:
    """Count already-assigned units that line up with unit_idx inside scope_idx."""
    box = text_boxes[unit_idx]
    h = max(1, box[3] - box[1])
    support = 0
    for j, assigned in enumerate(assignments):
        if assigned != scope_idx:
            continue
        other = text_boxes[j]
        oh = max(1, other[3] - other[1])
        min_h = max(1, min(h, oh))
        same_col = _x_overlap(box, other) > 0.45 and _gap_y(box, other) <= min_h * 1.4
        same_row = _y_overlap(box, other) > 0.55 and _gap_x(box, other) <= min_h * 1.8
        if same_col or same_row:
            support += 1
    return support


def assign_units_to_scopes(
    text_boxes: list[list[int]], scopes: list[BubbleScope],
) -> list[int | None]:
    """Assign each text unit to its best YOLO scope.

    Manga speech bubbles often overlap in YOLO boxes. First pass keeps confident
    assignments. Second pass resolves ambiguous overlap cases using neighboring
    text units already assigned to a scope, preventing true bubble lines from
    falling into the free-text path.
    """
    assignments: list[int | None] = []
    ambiguous_rows: list[tuple[int, list[tuple[int, float]]]] = []

    for i, tb in enumerate(text_boxes):
        candidates = []
        for j, s in enumerate(scopes):
            center = 1.0 if _center_inside(tb, s.bbox) else 0.0
            inside = _inside_ratio(tb, s.bbox)
            iou = _box_iou(tb, s.bbox)
            score = 0.70 * center + 0.25 * inside + 0.05 * iou
            if inside < 0.18 and not center:
                continue
            candidates.append((j, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        ambiguous = bool(best and second and best[1] - second[1] < 0.08 and second[1] > 0.35)
        if best is None or best[1] < 0.45:
            assignments.append(None)
        elif ambiguous:
            assignments.append(None)
            ambiguous_rows.append((i, candidates[:3]))
        else:
            assignments.append(best[0])

    # Resolve ambiguous units by local text continuity.
    for i, candidates in ambiguous_rows:
        scored = []
        for scope_idx, score in candidates:
            support = _neighbor_support(text_boxes, assignments, i, scope_idx)
            scored.append((support, score, scope_idx))
        scored.sort(reverse=True)
        best_support, best_score, best_scope = scored[0]
        second_support = scored[1][0] if len(scored) > 1 else -1
        if best_support > 0 and (best_support > second_support or best_score >= 0.55):
            assignments[i] = best_scope

    return assignments


def load_yolo_model(model_path: str | Path):
    """Load a YOLO model from a local .pt or .mlpackage path."""
    from ultralytics import YOLO

    p = Path(model_path)
    if p.suffix == ".mlpackage":
        return YOLO(str(p), task="detect")
    return YOLO(str(p))
