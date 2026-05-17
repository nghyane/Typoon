"""Single-page PreparedReader shim — feed scan_chapter from one image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from typoon.domain.prepared import Chapter as PreparedChapter
from typoon.domain.prepared import Page as PreparedPage


class SinglePageReader:
    """Mimic the slice of `PreparedReader` API the stages actually use:
    `page_count`, `read_rgb(index)`, `chapter(source=…)`, context-manager.
    """

    def __init__(self, image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB image, got shape {image.shape}")
        self._image = image
        h, w = image.shape[:2]
        self._page = PreparedPage(index=0, width=w, height=h)

    @classmethod
    def from_path(cls, path: Path) -> "SinglePageReader":
        return cls(np.asarray(Image.open(path).convert("RGB")))

    @property
    def page_count(self) -> int:
        return 1

    def read_rgb(self, index: int) -> np.ndarray:
        if index != 0:
            raise IndexError(f"single-page reader only has page 0; got {index}")
        return self._image

    def chapter(self, source: str = "", *, is_color: bool | None = None) -> PreparedChapter:
        if is_color is None:
            is_color = self._detect_is_color()
        return PreparedChapter(
            source=source or "probe",
            pages=(self._page,),
            is_color=is_color,
            strategy="one_to_one",
        )

    def close(self) -> None:
        pass

    def __enter__(self) -> "SinglePageReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # Mirror PreparedReader's HSV-saturation sample so brief gets the
    # same is_color signal the worker would compute.
    def _detect_is_color(self) -> bool:
        img = self._image
        small = cv2.resize(img, (256, 256)) if min(img.shape[:2]) > 256 else img
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        ratio = float(np.count_nonzero(hsv[:, :, 1] > 30)) / (hsv.shape[0] * hsv.shape[1])
        return ratio >= 0.15
