"""Page scanner — PP-OCR detection + OCR backend.

PP-OCR det runs on every page to find text units (passed to grouping).
OCR is a `PageOcr` (Apple Vision / Lens / Windows / Tesseract) called
once per page on the full image, or a `CropOcr` (manga-ocr) called
per group. The grouping pipeline routes between the two by checking
which protocol the backend implements.

The scanner holds the active source language so the OCR call gets the
right recogniser without threading `lang` through the grouping API.
"""

from __future__ import annotations

import numpy as np

from .ocr import CropOcr, PageOcr, create_ocr


class Scanner:
    def __init__(self, detector, ocr: PageOcr | CropOcr) -> None:
        self._det = detector
        self._ocr = ocr
        self._lang: str | None = None

    def set_language(self, lang: str | None) -> None:
        self._lang = lang

    @property
    def lang(self) -> str | None:
        return self._lang

    @property
    def ocr(self) -> PageOcr | CropOcr:
        return self._ocr

    def scan(self, image, *, scope_model=None, scope_imgsz=640, scope_conf=0.3):
        from typoon.vision.grouping import export_groups, scan_page
        return export_groups(scan_page(
            self, image,
            yolo_model=scope_model,
            yolo_imgsz=scope_imgsz,
            yolo_conf=scope_conf,
        ))


def create_scanner(
    hub=None,
    *,
    ocr_backend: str = "auto",
    source_lang: str | None = None,
    lens_endpoint: str | None = None,
) -> Scanner:
    """Build a Scanner with PP-OCR det + the configured OCR backend."""
    if hub is None:
        raise RuntimeError("PP-OCR models required")
    from .detect import TextDetector
    detector = TextDetector(
        hub.resolve("ppocr-det.safetensors"),
        hub.resolve("ppocr-det-config.json"),
    )
    ocr = create_ocr(source_lang, backend=ocr_backend, lens_endpoint=lens_endpoint)
    scanner = Scanner(detector, ocr)
    if source_lang is not None:
        scanner.set_language(source_lang)
    return scanner
