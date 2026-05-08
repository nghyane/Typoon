"""Visual inspection outputs for text detection, grouping, and masks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .draw import CYAN, GREEN, MAGENTA, PALETTE, RED, YELLOW, hstack, label, rect, write_rgb
from .types import ScanState


def write_inspection(out_dir: Path, page_index: int, image: np.ndarray, state: ScanState, eraser) -> None:
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


def state_to_dict(page_index: int, image_file: str, state: ScanState) -> dict[str, Any]:
    return {
        "page": page_index,
        "file": image_file,
        "width": state.width,
        "height": state.height,
        "units": [
            {
                "idx": u.idx,
                "bbox": u.bbox,
                "text": u.text,
                "confidence": u.confidence,
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
                "text": g.text,
                "confidence": g.confidence,
                "accepted": g.accepted,
                "reject_reason": g.reject_reason,
                "scoped": g.scoped,
                "scope_idx": g.scope_idx,
                "shape_kind": g.shape_kind,
            }
            for g in state.groups
        ],
    }


def _draw_text_boxes(image: np.ndarray, state: ScanState) -> np.ndarray:
    out = image.copy()
    for u in state.units:
        color = RED if u.is_noise else CYAN
        rect(out, u.bbox, color, 2)
        label(out, u.bbox[0], u.bbox[1], f"u{u.idx} {u.confidence:.2f}", color)
    return out


def _draw_groups(image: np.ndarray, state: ScanState) -> np.ndarray:
    out = image.copy()
    for s in state.scopes:
        rect(out, s.bbox, YELLOW, 2)
        label(out, s.bbox[0], s.bbox[1], f"s{s.idx} {s.confidence:.2f}", YELLOW)
    for g in state.groups:
        if not g.accepted:
            continue
        color = MAGENTA if g.shape_kind == "burst" else GREEN
        rect(out, g.fit_bbox, color, 3)
        text = g.text[:24].replace("\n", " ")
        tag = "BURST " if g.shape_kind == "burst" else ""
        label(out, g.fit_bbox[0], g.fit_bbox[1],
              f"g{g.idx} {tag}{g.confidence:.2f} {text}", color)
    return out


def _draw_masks(image: np.ndarray, state: ScanState) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    accepted = [g for g in state.groups if g.accepted]
    for gi, g in enumerate(accepted):
        color = PALETTE[gi % len(PALETTE)]
        for ui in g.unit_indices:
            mask = state.units[ui].region.mask
            if mask is None:
                continue
            h, w = mask.image.shape[:2]
            y1, y2 = max(0, mask.y), min(out.shape[0], mask.y + h)
            x1, x2 = max(0, mask.x), min(out.shape[1], mask.x + w)
            if y2 <= y1 or x2 <= x1:
                continue
            crop = mask.image[y1 - mask.y:y2 - mask.y, x1 - mask.x:x2 - mask.x]
            overlay[y1:y2, x1:x2][crop > 0] = color
        rect(out, g.fit_bbox, color, 2)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    for gi, g in enumerate(accepted):
        rect(out, g.fit_bbox, PALETTE[gi % len(PALETTE)], 2)
    return out


def _draw_erased(image: np.ndarray, state: ScanState, eraser) -> np.ndarray:
    from .grouping import export_groups
    all_masks = [m for g in export_groups(state) for m in g.erase_masks]
    if not all_masks:
        return image.copy()
    h, w = image.shape[:2]
    canvas = np.dstack([image, np.full((h, w), 255, dtype=np.uint8)])
    eraser.erase(canvas, all_masks)
    return canvas[:, :, :3]
