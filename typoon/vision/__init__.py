"""Vision pipeline — detection, OCR, grouping, and text erasure."""

from .erase import Eraser
from .grouping import export_groups, scan_page
from .types import DetectedGroup, ScanState, TextMask, TextRegion

__all__ = [
    "Eraser",
    "export_groups", "scan_page",
    "DetectedGroup", "ScanState", "TextMask", "TextRegion",
]
