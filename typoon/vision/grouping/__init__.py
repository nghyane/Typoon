"""Vision grouping pipeline — public API.

Entry points:
  scan_page(scanner, image, ...)  -> ScanState
  export_groups(state)            -> list[DetectedGroup]
  build_erase_masks(masks, ...)   -> list[TextMask]
"""

from __future__ import annotations

import numpy as np

from typoon.vision.types import DetectedGroup, ScanState

from .groups import build_erase_masks, build_groups, export_groups, filter_groups, ocr_groups
from .units import (
    assign_scopes,
    detect_scopes,
    detect_units,
    filter_units,
    split_units,
)

__all__ = [
    "scan_page",
    "export_groups",
    "build_erase_masks",
]


def scan_page(
    scanner,
    image: np.ndarray,
    *,
    yolo_model=None,
    yolo_imgsz: int = 640,
    yolo_conf: float = 0.3,
) -> ScanState:
    """Run full grouping pipeline: detect → filter → scope → group → OCR → filter."""
    h, w = image.shape[:2]
    state = ScanState(image=image, width=w, height=h)
    detect_units(state, scanner)
    if not state.units:
        return state
    filter_units(state)
    detect_scopes(state, yolo_model, yolo_imgsz, yolo_conf)
    split_units(state)
    assign_scopes(state)
    build_groups(state)
    ocr_groups(state, scanner)
    filter_groups(state)
    return state


