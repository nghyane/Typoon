"""Vision grouping pipeline — public API.

Entry points:
  build_page_scan_state(scanner, image, ...) -> PageScanState
  to_visual_text_groups(state)              -> list[VisualTextGroup]
"""

from __future__ import annotations

import numpy as np

from typoon.vision.types import PageScanState, VisualTextGroup

from .groups import (
    build_erase_masks,
    build_groups,
    final_filter_groups,
    ocr_groups,
    to_visual_text_groups,
)
from .units import (
    assign_units_to_scopes,
    detect_scopes,
    detect_units,
    filter_units,
    ocr_units_for_filtering,
    split_units_crossing_scopes,
)

__all__ = [
    "build_page_scan_state",
    "to_visual_text_groups",
    "build_erase_masks",
]


def build_page_scan_state(
    scanner,
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
    split_units_crossing_scopes(state)
    assign_units_to_scopes(state)
    build_groups(state)
    ocr_groups(state, scanner)
    final_filter_groups(state)
    return state
