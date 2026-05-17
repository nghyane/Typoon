"""Compose four panels into a single 2×2 overview image."""

from __future__ import annotations

import numpy as np


def grid_2x2(
    tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray,
    *, gap: int = 8, bg: tuple[int, int, int] = (24, 24, 24),
) -> np.ndarray:
    """Stack four equally-sized panels into one image with a gap separator."""
    assert tl.shape == tr.shape == bl.shape == br.shape, "panels must match shape"
    H, W = tl.shape[:2]
    out = np.full(
        (H * 2 + gap, W * 2 + gap, 3), bg, dtype=np.uint8,
    )
    out[:H,        :W]            = tl
    out[:H,        W + gap:]      = tr
    out[H + gap:,  :W]            = bl
    out[H + gap:,  W + gap:]      = br
    return out
