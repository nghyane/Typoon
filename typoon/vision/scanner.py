"""Page scanner — PP-OCR detection + pluggable OcrBackend.

Detection: PP-OCR det (DBNet++), shared across all platforms.
Recognition: injected OcrBackend (Apple Vision / Windows / Tesseract).
"""

from __future__ import annotations

import numpy as np

from .ocr_backend import OcrBackend, create_ocr_backend
from .types import VisualTextGroup


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

    def scan(
        self,
        image: np.ndarray,
        *,
        scope_model=None,
        scope_imgsz: int = 640,
        scope_conf: float = 0.3,
    ) -> list[VisualTextGroup]:
        from .grouping import build_page_scan_state, to_visual_text_groups
        state = build_page_scan_state(
            self, image,
            yolo_model=scope_model,
            yolo_imgsz=scope_imgsz,
            yolo_conf=scope_conf,
        )
        return to_visual_text_groups(state)


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
