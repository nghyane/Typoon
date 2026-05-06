"""PreparedReader — decode RGB ndarrays from a Bunle prepared archive.

Stage code receives (PreparedChapter, PreparedReader) instead of reading
files via `chapter.page_path(...)`. The reader is mmap-backed; pages are
decoded on demand and not cached.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import bunle
import numpy as np
from PIL import Image

from typoon.domain.prepared import Chapter, Page


class PreparedReader:
    """Random-access reader for a prepared Bunle archive.

    Use as a context manager:

        with PreparedReader.open(archive_path) as reader:
            chapter = reader.chapter()
            for i in range(chapter.page_count):
                rgb = reader.read_rgb(i)
    """

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

    def chapter(self, source: str = "") -> Chapter:
        pages = tuple(
            Page(
                index=i,
                width=self._reader.info(i)["width"],
                height=self._reader.info(i)["height"],
                file="",  # archive-backed: no on-disk file
            )
            for i in range(self._reader.page_count)
        )
        return Chapter(root=Path(self._archive_path), source=source, pages=pages)

    def read_rgb(self, index: int) -> np.ndarray:
        data = self._reader.page(index)
        with Image.open(BytesIO(data)) as img:
            return np.asarray(img.convert("RGB"))
