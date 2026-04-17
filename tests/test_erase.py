"""Tests for text erasure."""

from __future__ import annotations

import numpy as np
import pytest

from typoon.vision.erase import Eraser, _cluster_masks, _erase_with_median, _sample_background
from typoon.vision.types import TextMask
from .conftest import MODELS_DIR, make_mask, skip_no_lama


def _white_canvas(w: int = 400, h: int = 300) -> np.ndarray:
    return np.full((h, w, 4), 255, dtype=np.uint8)


def test_median_erase_flat_bg():
    canvas = _white_canvas()
    canvas[120:180, 100:300] = [0, 0, 0, 255]
    mask = make_mask(100, 120, 200, 60)
    _erase_with_median(canvas, mask)
    assert canvas[150, 200, 0] == 255


def test_cluster_nearby():
    m1 = make_mask(100, 100, 50, 30)
    m2 = make_mask(110, 135, 50, 30)
    assert len(_cluster_masks([m1, m2])) == 1


def test_cluster_distant():
    m1 = make_mask(0, 0, 10, 10)
    m2 = make_mask(200, 200, 10, 10)
    assert len(_cluster_masks([m1, m2])) == 2


def test_sample_background_white():
    canvas = _white_canvas()
    mask = make_mask(10, 10, 50, 50, fill=0)
    color, spread = _sample_background(canvas, mask)
    assert color[0] == 255
    assert spread == 0


def test_eraser_median_fallback():
    canvas = _white_canvas()
    canvas[120:180, 100:300] = [0, 0, 0, 255]
    mask = make_mask(100, 120, 200, 60)
    eraser = Eraser()
    eraser.erase(canvas, [mask])
    assert canvas[150, 200, 0] == 255


@skip_no_lama
def test_eraser_lama():
    canvas = _white_canvas(w=400, h=300)
    # Textured background that extends INTO the mask bbox
    # so non-masked pixels inside bbox are non-uniform → triggers LaMa
    canvas[100:200, 80:320, :3] = np.random.randint(100, 200, (100, 240, 3), dtype=np.uint8)
    # Mask covers only center stripe — leaves textured pixels at top/bottom of bbox
    mask = make_mask(100, 130, 200, 30)
    eraser = Eraser(model_path=str(MODELS_DIR / "lama-manga.safetensors"))
    eraser.erase(canvas, [mask])
    assert canvas[145, 200, 0] != 0
