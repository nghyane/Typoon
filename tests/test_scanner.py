"""Tests for page scanners (VisionScanner + TesseractScanner)."""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from typoon.vision.scanner import (
    ScannedBubble,
    VisionScanner,
    TesseractScanner,
    create_scanner,
    _vision_available,
    _tesseract_available,
)
from .conftest import FIXTURES_DIR, MODELS_DIR, skip_no_ppocr_det


# ── Fixtures ─────────────────────────────────────────────────────

def _load_page(name: str = "ch013/03.webp") -> np.ndarray:
    path = FIXTURES_DIR / "ctrlaltresign" / name
    if not path.exists():
        pytest.skip(f"fixture not found: {path}")
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


skip_no_vision = pytest.mark.skipif(
    not _vision_available(),
    reason="macOS Vision framework not available",
)


def _make_vision_scanner(languages=None):
    """Create VisionScanner with PP-OCR detector (required since hybrid rework)."""
    from typoon.vision.detect import TextDetector
    det_path = MODELS_DIR / "ppocr-det.safetensors"
    cfg_path = MODELS_DIR / "ppocr-det-config.json"
    if not det_path.exists():
        pytest.skip("PP-OCR det model not found")
    return VisionScanner(
        detector=TextDetector(str(det_path), str(cfg_path)),
        languages=languages or ["en-US"],
    )


# ── ScannedBubble ────────────────────────────────────────────────

def test_scanned_bubble_fields():
    b = ScannedBubble(
        polygon=[[0, 0], [100, 0], [100, 50], [0, 50]],
        text="hello",
        confidence=0.95,
    )
    assert b.text == "hello"
    assert b.confidence == 0.95
    assert b.masks == []


# ── create_scanner ───────────────────────────────────────────────

def test_create_scanner_needs_hub():
    """Hub is always required (PP-OCR det used by all scanners)."""
    with pytest.raises(RuntimeError, match="PP-OCR models required"):
        create_scanner(hub=None)


# ── VisionScanner ────────────────────────────────────────────────

@skip_no_vision
class TestVisionScanner:

    def test_scan_returns_bubbles(self):
        img = _load_page()
        scanner = _make_vision_scanner()
        bubbles = scanner.scan(img)
        assert len(bubbles) > 0
        for b in bubbles:
            assert isinstance(b, ScannedBubble)
            assert len(b.polygon) >= 4
            assert len(b.text) > 0
            assert b.confidence > 0

    def test_scan_text_quality(self):
        """Vision should produce high-confidence readable text."""
        img = _load_page()
        scanner = _make_vision_scanner()
        bubbles = scanner.scan(img)
        high_conf = [b for b in bubbles if b.confidence >= 0.8]
        # At least half of bubbles should be high confidence
        assert len(high_conf) >= len(bubbles) // 2

    def test_scan_empty_image(self):
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        scanner = _make_vision_scanner()
        bubbles = scanner.scan(img)
        assert bubbles == []

    def test_scan_polygon_coords_in_bounds(self):
        img = _load_page()
        h, w = img.shape[:2]
        scanner = _make_vision_scanner()
        bubbles = scanner.scan(img)
        for b in bubbles:
            for p in b.polygon:
                assert 0 <= p[0] <= w + 1, f"x={p[0]} out of bounds (w={w})"
                assert 0 <= p[1] <= h + 1, f"y={p[1]} out of bounds (h={h})"

    def test_scan_multiple_pages_consistent(self):
        """Same image should produce same results."""
        img = _load_page()
        scanner = _make_vision_scanner()
        b1 = scanner.scan(img)
        b2 = scanner.scan(img)
        assert len(b1) == len(b2)
        for a, b in zip(b1, b2):
            assert a.text == b.text

    def test_scan_manga_page(self):
        """Test on actual manga page with known content."""
        path = FIXTURES_DIR / "ctrlaltresign" / "ch013" / "14.webp"
        if not path.exists():
            pytest.skip("fixture not found")
        img = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scanner = _make_vision_scanner()
        bubbles = scanner.scan(img_rgb)
        # Should detect some text bubbles
        assert len(bubbles) >= 1
        # All bubbles should have non-empty text
        assert all(b.text.strip() for b in bubbles)


# ── TesseractScanner ──────────────────────────────────────────────

skip_no_tesseract = pytest.mark.skipif(
    not _tesseract_available(),
    reason="Tesseract not available",
)


def _make_tesseract_scanner():
    from typoon.vision.detect import TextDetector
    det_path = MODELS_DIR / "ppocr-det.safetensors"
    if not det_path.exists():
        pytest.skip("PP-OCR det model not found")
    return TesseractScanner(
        detector=TextDetector(str(det_path), str(MODELS_DIR / "ppocr-det-config.json")),
    )


@skip_no_tesseract
@skip_no_ppocr_det
class TestTesseractScanner:

    def test_scan_returns_bubbles(self):
        img = _load_page()
        scanner = _make_tesseract_scanner()
        bubbles = scanner.scan(img)
        assert len(bubbles) > 0
        for b in bubbles:
            assert isinstance(b, ScannedBubble)
            assert len(b.polygon) >= 4
            assert len(b.text) > 0

    def test_scan_empty_image(self):
        img = np.full((200, 300, 3), 255, dtype=np.uint8)
        scanner = _make_tesseract_scanner()
        bubbles = scanner.scan(img)
        assert bubbles == []


# ── Cross-scanner comparison ─────────────────────────────────────

@skip_no_vision
@skip_no_tesseract
@skip_no_ppocr_det
def test_vision_vs_tesseract_bubble_count():
    """Vision and Tesseract should detect similar number of bubbles (same det)."""
    img = _load_page()
    v_bubbles = _make_vision_scanner().scan(img)
    t_bubbles = _make_tesseract_scanner().scan(img)
    assert len(v_bubbles) > 0
    assert len(t_bubbles) > 0
    ratio = len(v_bubbles) / max(len(t_bubbles), 1)
    assert 0.3 < ratio < 3.0, f"Vision={len(v_bubbles)} vs Tesseract={len(t_bubbles)}"
