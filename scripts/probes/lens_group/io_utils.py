"""Image I/O + JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    """Decode any PIL-supported format to contiguous uint8 RGB (H, W, 3)."""
    return np.asarray(Image.open(path).convert("RGB"))


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
