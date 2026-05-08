"""Page scanner — PP-OCR detection + pluggable OcrBackend.

Detection: PP-OCR det (DBNet++), shared across all platforms.
Recognition: injected OcrBackend (Apple Vision / Windows / Tesseract).
"""

from __future__ import annotations

import numpy as np

from .ocr_backend import OcrBackend, create_ocr_backend


class Scanner:
    """Combines PP-OCR detector with an OcrBackend.

    Implements the _GroupingScanner protocol expected by grouping pipeline:
    exposes ._det and ._ocr_crops().
    """

    def __init__(self, detector, ocr: OcrBackend) -> None:
        self._det = detector
        self._ocr = ocr

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        return self._ocr.recognize(crops)

    def scan(self, image, *, scope_model=None, scope_imgsz=640, scope_conf=0.3):
        from typoon.vision.grouping import export_groups, scan_page
        return export_groups(scan_page(self, image, yolo_model=scope_model,
                                       yolo_imgsz=scope_imgsz, yolo_conf=scope_conf))


def create_scanner(hub=None, languages: list[str] | None = None) -> Scanner:
    """Create Scanner with best available OCR backend for this platform."""
    if hub is None:
        raise RuntimeError("PP-OCR models required")
    from .detect import TextDetector
    detector = TextDetector(
        hub.resolve("ppocr-det.safetensors"),
        hub.resolve("ppocr-det-config.json"),
    )
    ocr = create_ocr_backend(languages)
    return Scanner(detector, ocr)
