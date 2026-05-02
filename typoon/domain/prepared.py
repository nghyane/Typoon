"""PreparedChapter — pages directory is the source of truth."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typoon.paths import ChapterPaths


@dataclass(frozen=True)
class Page:
    index:  int
    width:  int
    height: int
    file:   str     # relative to chapter root, e.g. "pages/0000.png"


@dataclass(frozen=True)
class Chapter:
    root:   Path
    source: str
    pages:  tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page_path(self, index: int) -> Path:
        return self.root / self.pages[index].file

    @classmethod
    def from_paths(cls, cp: "ChapterPaths", source: str = "") -> "Chapter":
        """Build Chapter by scanning the pages directory.
        Reads PNG dimensions from header only — does not decode pixel data.
        """
        pages = []
        for png in sorted(cp.pages.glob("*.png")):
            w, h = _png_dimensions(png)
            index = int(png.stem)
            pages.append(Page(index=index, width=w, height=h, file=f"pages/{png.name}"))
        return cls(root=cp.root, source=source, pages=tuple(pages))


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read width, height from PNG IHDR — no pixel decoding."""
    with open(path, "rb") as f:
        f.read(8)           # PNG signature
        f.read(4)           # IHDR chunk length
        f.read(4)           # 'IHDR'
        w = struct.unpack(">I", f.read(4))[0]
        h = struct.unpack(">I", f.read(4))[0]
    return w, h
