"""PreparedChapter — pages directory is the source of truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from typoon.paths import ChapterPaths

_PAGE_EXTS = ("*.png", "*.webp", "*.jpg", "*.jpeg", "*.avif", "*.tiff", "*.bmp")


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
        """Build Chapter by scanning the pages directory. Supports all image formats."""
        files = sorted(
            {p for ext in _PAGE_EXTS for p in cp.pages.glob(ext)},
            key=lambda p: p.stem,
        )
        pages = []
        for f in files:
            with Image.open(f) as img:
                w, h = img.size
            index = int(f.stem)
            pages.append(Page(index=index, width=w, height=h, file=f"pages/{f.name}"))
        return cls(root=cp.root, source=source, pages=tuple(pages))
