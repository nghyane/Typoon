"""Backward-compat shim — all symbols forwarded to vision/grouping/."""

from typoon.vision.grouping import scan_page as build_page_scan_state, export_groups as to_visual_text_groups, build_erase_masks
from typoon.vision.grouping.geometry import poly_bbox as bbox, union_boxes, pad_box as expand, box_to_poly as box_to_polygon, fit_padding
from typoon.vision.grouping.units import detect_units, filter_units, ocr_units as ocr_units_for_filtering
from typoon.vision.grouping.units import detect_scopes, assign_scopes as assign_units_to_scopes, split_units as split_units_crossing_scopes
from typoon.vision.grouping.groups import build_groups, ocr_groups, filter_groups as final_filter_groups, subgroup_blocks as subgroup_text_blocks

__all__ = [
    "build_page_scan_state", "to_visual_text_groups", "build_erase_masks",
    "bbox", "union_boxes", "expand", "box_to_polygon", "fit_padding",
    "detect_units", "filter_units", "ocr_units_for_filtering",
    "detect_scopes", "assign_units_to_scopes", "split_units_crossing_scopes",
    "build_groups", "ocr_groups", "final_filter_groups", "subgroup_text_blocks",
]


def group_and_ocr(scanner, image, *, yolo_model=None, yolo_imgsz=640, yolo_conf=0.3):
    from typoon.vision.grouping import scan_page, export_groups
    return export_groups(scan_page(scanner, image, yolo_model=yolo_model,
                                   yolo_imgsz=yolo_imgsz, yolo_conf=yolo_conf))
