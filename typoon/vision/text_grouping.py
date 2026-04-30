"""Backward-compat shim — imports forwarded to vision/grouping/.

New code should import from typoon.vision.grouping directly.
"""

from typoon.vision.grouping import build_page_scan_state, to_visual_text_groups, build_erase_masks
from typoon.vision.grouping.geometry import bbox, union_boxes, expand, box_to_polygon, fit_padding
from typoon.vision.grouping.units import detect_units, filter_units, ocr_units_for_filtering
from typoon.vision.grouping.units import detect_scopes, assign_units_to_scopes, split_units_crossing_scopes
from typoon.vision.grouping.groups import build_groups, ocr_groups, final_filter_groups, subgroup_text_blocks

__all__ = [
    "build_page_scan_state", "to_visual_text_groups", "build_erase_masks",
    "bbox", "union_boxes", "expand", "box_to_polygon", "fit_padding",
    "detect_units", "filter_units", "ocr_units_for_filtering",
    "detect_scopes", "assign_units_to_scopes", "split_units_crossing_scopes",
    "build_groups", "ocr_groups", "final_filter_groups", "subgroup_text_blocks",
]

# Legacy alias used by _BaseScanner.scan()
def group_and_ocr(scanner, image, *, yolo_model=None, yolo_imgsz=640, yolo_conf=0.3):
    state = build_page_scan_state(scanner, image,
                                   yolo_model=yolo_model,
                                   yolo_imgsz=yolo_imgsz,
                                   yolo_conf=yolo_conf)
    return to_visual_text_groups(state)
