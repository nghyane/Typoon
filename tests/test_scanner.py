"""End-to-end scanner tests (detect → merge → OCR).

Unit-testing the PP-OCR detector directly is not useful — it's a thin
adapter over CoreML/MLX/ONNX. The scanner pipeline is where our logic
lives, so we exercise it through real fixtures.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from typoon.vision.scanner import (
    ScannedBubble,
    TesseractScanner,
    VisionScanner,
    _tesseract_available,
    _vision_available,
    create_scanner,
)
from .conftest import FIXTURES_DIR, MODELS_DIR, skip_no_ppocr_det


# ── Fixtures ─────────────────────────────────────────────────────

def _load_page(name: str = "ch013/03.webp") -> np.ndarray:
    path = FIXTURES_DIR / "ctrlaltresign" / name
    if not path.exists():
        pytest.skip(f"fixture not found: {path}")
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _make_scanner(cls):
    from typoon.vision.detect import TextDetector
    det_path = MODELS_DIR / "ppocr-det.safetensors"
    cfg_path = MODELS_DIR / "ppocr-det-config.json"
    if not det_path.exists():
        pytest.skip("PP-OCR det model not found")
    detector = TextDetector(str(det_path), str(cfg_path))
    return cls(detector=detector) if cls is TesseractScanner else cls(detector=detector, languages=["en-US"])


skip_no_vision = pytest.mark.skipif(not _vision_available(), reason="macOS Vision not available")
skip_no_tesseract = pytest.mark.skipif(not _tesseract_available(), reason="Tesseract not available")


# ── Types + factory ──────────────────────────────────────────────

def test_scanned_bubble_defaults():
    b = ScannedBubble(polygon=[[0, 0], [100, 0], [100, 50], [0, 50]], text="hi", confidence=0.9)
    assert b.text == "hi"
    assert b.confidence == 0.9
    assert b.masks == []


def test_create_scanner_needs_hub():
    with pytest.raises(RuntimeError, match="PP-OCR models required"):
        create_scanner(hub=None)


# ── End-to-end ───────────────────────────────────────────────────

@skip_no_vision
@skip_no_ppocr_det
class TestVisionScanner:

    def test_scan_manga_page(self):
        """Full pipeline on a real manga page — tiling, detect, merge, OCR."""
        img = _load_page("ch013/14.webp")
        bubbles = _make_scanner(VisionScanner).scan(img)
        assert len(bubbles) >= 1
        for b in bubbles:
            assert isinstance(b, ScannedBubble)
            assert len(b.polygon) >= 4
            assert b.text.strip()
            assert b.confidence > 0
            h, w = img.shape[:2]
            for p in b.polygon:
                assert 0 <= p[0] <= w + 1
                assert 0 <= p[1] <= h + 1

    def test_scan_empty_image(self):
        assert _make_scanner(VisionScanner).scan(np.full((200, 300, 3), 255, dtype=np.uint8)) == []


@skip_no_tesseract
@skip_no_ppocr_det
def test_tesseract_scanner_smoke():
    """Tesseract path is a fallback — smoke test only."""
    img = _load_page()
    bubbles = _make_scanner(TesseractScanner).scan(img)
    assert len(bubbles) > 0
    assert all(b.text for b in bubbles)
