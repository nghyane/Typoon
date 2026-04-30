"""Visual inspection outputs for text detection, grouping, and masks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .draw import CYAN, GREEN, PALETTE, RED, YELLOW, hstack, label, rect, write_rgb
from .types import PageScanState


def write_inspection(out_dir: Path, page_index: int, image: np.ndarray, state: PageScanState, eraser) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        _draw_text_boxes(image, state),
        _draw_groups(image, state),
        _draw_masks(image, state),
        _draw_erased(image, state, eraser),
    ]
    write_rgb(out_dir / f"page_{page_index:04d}.png", hstack(panels))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", "utf-8")


def state_to_dict(page_index: int, image_file: str, state: PageScanState) -> dict[str, Any]:
    return {
        "page": page_index,
        "file": image_file,
        "width": state.width,
        "height": state.height,
        "units": [
            {
                "idx": u.idx,
                "bbox": u.bbox,
                "text": u.unit_ocr_text,
                "confidence": u.unit_ocr_conf,
                "noise": u.is_noise,
                "noise_reason": u.noise_reason,
                "scope_idx": u.scope_idx,
            }
            for u in state.units
        ],
        "scopes": [
            {"idx": s.idx, "bbox": s.bbox, "confidence": s.confidence}
            for s in state.scopes
        ],
        "groups": [
            {
                "idx": g.idx,
                "unit_indices": g.unit_indices,
                "raw_bbox": g.raw_bbox,
                "fit_bbox": g.fit_bbox,
                "ocr_bbox": g.ocr_bbox,
                "text": g.ocr_text,
                "confidence": g.ocr_conf,
                "accepted": g.accepted,
                "reject_reason": g.reject_reason,
                "scoped": g.scoped,
                "scope_idx": g.scope_idx,
            }
            for g in state.groups
        ],
    }


def _draw_text_boxes(image: np.ndarray, state: PageScanState) -> np.ndarray:
    out = image.copy()
    for unit in state.units:
        color = RED if unit.is_noise else CYAN
        rect(out, unit.bbox, color, 2)
        label(out, unit.bbox[0], unit.bbox[1], f"u{unit.idx} {unit.unit_ocr_conf:.2f}", color)
    return out


def _draw_groups(image: np.ndarray, state: PageScanState) -> np.ndarray:
    out = image.copy()
    for scope in state.scopes:
        rect(out, scope.bbox, YELLOW, 2)
        label(out, scope.bbox[0], scope.bbox[1], f"s{scope.idx} {scope.confidence:.2f}", YELLOW)
    for group in state.groups:
        if not group.accepted:
            continue
        rect(out, group.fit_bbox, GREEN, 3)
        text = group.ocr_text[:24].replace("\n", " ")
        label(out, group.fit_bbox[0], group.fit_bbox[1], f"g{group.idx} {group.ocr_conf:.2f} {text}", GREEN)
    return out


def _draw_masks(image: np.ndarray, state: PageScanState) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    accepted = [g for g in state.groups if g.accepted]
    for gi, group in enumerate(accepted):
        color = PALETTE[gi % len(PALETTE)]
        for unit_idx in group.unit_indices:
            mask = state.units[unit_idx].region.mask
            if mask is None:
                continue
            h, w = mask.image.shape[:2]
            y1, y2 = max(0, mask.y), min(out.shape[0], mask.y + h)
            x1, x2 = max(0, mask.x), min(out.shape[1], mask.x + w)
            if y2 <= y1 or x2 <= x1:
                continue
            crop = mask.image[y1 - mask.y:y2 - mask.y, x1 - mask.x:x2 - mask.x]
            overlay[y1:y2, x1:x2][crop > 0] = color
        rect(out, group.fit_bbox, color, 2)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for gi, group in enumerate(accepted):
        rect(out, group.fit_bbox, PALETTE[gi % len(PALETTE)], 2)
    return out


def _draw_erased(image: np.ndarray, state: PageScanState, eraser) -> np.ndarray:
    from .text_grouping import to_visual_text_groups
    groups = to_visual_text_groups(state)
    all_masks = [m for g in groups for m in g.erase_masks]
    if not all_masks:
        return image.copy()
    h, w = image.shape[:2]
    canvas = np.dstack([image, np.full((h, w), 255, dtype=np.uint8)])
    eraser.erase(canvas, all_masks)
    return canvas[:, :, :3]
