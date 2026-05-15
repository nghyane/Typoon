"""PreparedReader — decode RGB ndarrays from a Bunle prepared archive.

Stage code receives `(PreparedChapter, PreparedReader)` together. The
chapter carries metadata (page count + dimensions); the reader provides
random-access pixel decode backed by mmap.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import bunle
import cv2
import numpy as np
from PIL import Image

from typoon.domain.prepared import Chapter, Page


class PreparedReader:
    """Random-access reader for a prepared Bunle archive."""

    def __init__(self, archive_path: Path, reader: bunle.Reader) -> None:
        self._archive_path = archive_path
        self._reader = reader

    @classmethod
    def open(cls, archive_path: Path) -> "PreparedReader":
        return cls(archive_path, bunle.Reader(str(archive_path)))

    def __enter__(self) -> "PreparedReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._reader.close()

    @property
    def page_count(self) -> int:
        return self._reader.page_count

    def chapter(self, source: str = "", *, is_color: bool | None = None) -> Chapter:
        pages = tuple(
            Page(
                index=i,
                width=self._reader.info(i)["width"],
                height=self._reader.info(i)["height"],
            )
            for i in range(self._reader.page_count)
        )
        if is_color is None:
            is_color = self._detect_is_color()
        return Chapter(source=source, pages=pages, is_color=is_color)

    # HSV saturation sample on the middle page. Matches the threshold used
    # by stages.prepare (_COLOR_RATIO_THRESHOLD=0.15, _SAT_THRESHOLD=30) so
    # archives that round-trip through PreparedReader recover the same flag
    # prepare_chapter originally wrote.
    def _detect_is_color(self) -> bool:
        n = self._reader.page_count
        if n == 0:
            return False
        img = self.read_rgb(n // 2)
        small = cv2.resize(img, (256, 256)) if min(img.shape[:2]) > 256 else img
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        ratio = float(np.count_nonzero(hsv[:, :, 1] > 30)) / (hsv.shape[0] * hsv.shape[1])
        return ratio >= 0.15

    def read_rgb(self, index: int) -> np.ndarray:
        data = self._reader.page(index)
        with Image.open(BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))
