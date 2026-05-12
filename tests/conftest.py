"""Shared fixtures for vision pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from typoon.vision.types import TextMask, TextRegion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def models_available(*names: str) -> bool:
    return all((MODELS_DIR / n).exists() for n in names)


skip_no_ppocr_det = pytest.mark.skipif(
    not models_available("ppocr-det.safetensors", "ppocr-det-config.json"),
    reason="ppocr-det.safetensors not found",
)
skip_no_lama = pytest.mark.skipif(
    not models_available("lama-manga.safetensors"),
    reason="lama-manga.safetensors not found",
)

_DUMMY_CROP = np.zeros((1, 1, 3), dtype=np.uint8)


def make_line(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> TextRegion:
    return TextRegion(
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        crop=_DUMMY_CROP,
        confidence=conf,
        mask=None,
    )


def make_mask(x: int, y: int, w: int, h: int, fill: int = 255) -> TextMask:
    img = np.full((h, w), fill, dtype=np.uint8)
    return TextMask(x=x, y=y, image=img)
