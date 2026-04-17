"""Tests for text detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from typoon.vision.detect import TextDetector
from .conftest import FIXTURES_DIR, MODELS_DIR, skip_no_ppocr_det


def _load_test_image() -> np.ndarray:
    path = FIXTURES_DIR / "ctrlaltresign" / "ch013" / "03.webp"
    if not path.exists():
        pytest.skip(f"fixture not found: {path}")
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@skip_no_ppocr_det
def test_ppocr_detection():
    img = _load_test_image()
    det = TextDetector(
        model_path=str(MODELS_DIR / "ppocr-det.safetensors"),
        config_path=str(MODELS_DIR / "ppocr-det-config.json"),
    )
    result = det.detect(img)
    assert len(result.regions) > 5
    assert result.prob_image is not None
    assert result.prob_image.shape[:2] == img.shape[:2]
    for r in result.regions:
        assert len(r.polygon) >= 4
        assert r.crop.shape[0] > 0 and r.crop.shape[1] > 0
        assert r.confidence > 0
