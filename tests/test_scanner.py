"""End-to-end scanner tests (detect → group → OCR)."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from typoon.vision.ocr_backend import (
    AppleVisionBackend,
    TesseractBackend,
    _tesseract_available,
    _vision_available,
)
from typoon.vision.scanner import Scanner, create_scanner
from typoon.vision.types import VisualTextGroup
from .conftest import FIXTURES_DIR, MODELS_DIR, skip_no_ppocr_det


def _load_page(name: str = "ch013/03.webp") -> np.ndarray:
    path = FIXTURES_DIR / "ctrlaltresign" / name
    if not path.exists():
        pytest.skip(f"fixture not found: {path}")
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _make_scanner(ocr_cls, **ocr_kwargs) -> Scanner:
    from typoon.vision.detect import TextDetector
    det_path = MODELS_DIR / "ppocr-det.safetensors"
    cfg_path = MODELS_DIR / "ppocr-det-config.json"
    if not det_path.exists():
        pytest.skip("PP-OCR det model not found")
    detector = TextDetector(str(det_path), str(cfg_path))
    return Scanner(detector=detector, ocr=ocr_cls(**ocr_kwargs))


skip_no_vision = pytest.mark.skipif(not _vision_available(), reason="macOS Vision not available")
skip_no_tesseract = pytest.mark.skipif(not _tesseract_available(), reason="Tesseract not available")


def test_scanned_bubble_defaults():
    polygon = [[0, 0], [100, 0], [100, 50], [0, 50]]
    b = VisualTextGroup(
        text="hi", confidence=0.9,
        text_polygon=polygon, render_polygon=polygon,
        text_bbox=[0, 0, 100, 50], mask_bbox=[0, 0, 100, 50],
        fit_bbox=[0, 0, 100, 50], erase_bbox=[0, 0, 100, 50],
    )
    assert b.text == "hi"
    assert b.confidence == 0.9
    assert b.erase_masks == []


def test_create_scanner_needs_hub():
    with pytest.raises(RuntimeError, match="PP-OCR models required"):
        create_scanner(hub=None)


@skip_no_vision
@skip_no_ppocr_det
class TestVisionScanner:

    def test_scan_manga_page(self):
        img = _load_page("ch013/14.webp")
        bubbles = _make_scanner(AppleVisionBackend, languages=["en-US"]).scan(img)
        assert len(bubbles) >= 1
        for b in bubbles:
            assert isinstance(b, VisualTextGroup)
            assert len(b.render_polygon) >= 4
            assert b.text.strip()
            assert b.confidence > 0
            h, w = img.shape[:2]
            for p in b.render_polygon:
                assert 0 <= p[0] <= w + 1
                assert 0 <= p[1] <= h + 1

    def test_scan_empty_image(self):
        assert _make_scanner(AppleVisionBackend, languages=["en-US"]).scan(
            np.full((200, 300, 3), 255, dtype=np.uint8)
        ) == []


@skip_no_tesseract
@skip_no_ppocr_det
def test_tesseract_scanner_smoke():
    img = _load_page()
    bubbles = _make_scanner(TesseractBackend).scan(img)
    assert len(bubbles) > 0
    assert all(b.text for b in bubbles)
