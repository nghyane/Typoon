"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


def models_available(*names: str) -> bool:
    return all((MODELS_DIR / n).exists() for n in names)


skip_no_ppocr_det = pytest.mark.skipif(
    not models_available("ppocr-det.safetensors", "ppocr-det-config.json"),
    reason="ppocr-det.safetensors not found",
)
