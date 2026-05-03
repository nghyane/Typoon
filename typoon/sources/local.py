"""Local folder chapter source — lazy per-page loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from typoon.sources.constants import IMAGE_EXTS


class LocalSource:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._files: list[Path] | None = None

    def _discover(self) -> list[Path]:
        if self._files is None:
            self._files = sorted(
                f for f in self._path.iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )
            if not self._files:
                raise FileNotFoundError(f"No images found in {self._path}")
        return self._files

    async def fetch(self) -> None:
        self._discover()

    def page_count(self) -> int:
        return len(self._discover())

    def load_page(self, index: int) -> np.ndarray:
        files = self._discover()
        path = files[index]
        # Use PIL to respect EXIF orientation before converting to array.
        # cv2.imread ignores EXIF orientation tags.
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            return np.array(rgb, dtype=np.uint8)
