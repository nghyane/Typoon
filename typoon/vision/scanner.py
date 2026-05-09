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

    Source language is set via `set_language` before each scan; one
    Scanner instance is shared across projects, so the recognizer
    language must be reset per call.
    """

    def __init__(self, detector, ocr: OcrBackend) -> None:
        self._det = detector
        self._ocr = ocr
        self._lang: str | None = None

    def set_language(self, lang: str | None) -> None:
        self._lang = lang

    @property
    def wants_raw(self) -> bool:
        """Skip pipeline binarization if the active backend prefers raw RGB.

        `OcrBackend.wants_raw` may be either a bool attribute (per-backend
        flag) or a callable that resolves the flag against the active
        language (RoutingBackend).
        """
        flag = getattr(self._ocr, "wants_raw", False)
        return bool(flag(self._lang)) if callable(flag) else bool(flag)

    def _ocr_crops(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        return self._ocr.recognize(crops, lang=self._lang)

    def scan(self, image, *, scope_model=None, scope_imgsz=640, scope_conf=0.3):
        from typoon.vision.grouping import export_groups, scan_page
        return export_groups(scan_page(self, image, yolo_model=scope_model,
                                       yolo_imgsz=scope_imgsz, yolo_conf=scope_conf))


def create_scanner(hub=None) -> Scanner:
    """Create Scanner with best available OCR backend for this platform."""
    if hub is None:
        raise RuntimeError("PP-OCR models required")
    from .detect import TextDetector
    detector = TextDetector(
        hub.resolve("ppocr-det.safetensors"),
        hub.resolve("ppocr-det-config.json"),
    )
    ocr = create_ocr_backend()
    return Scanner(detector, ocr)
