"""PreparedChapter manifest contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreparedPage:
    index: int
    file: str
    width: int
    height: int


@dataclass(frozen=True)
class PreparedChapter:
    root: Path
    source: str
    pages: tuple[PreparedPage, ...]
    version: int = 1

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page_path(self, index: int) -> Path:
        return self.root / self.pages[index].file

    def to_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "page_count": self.page_count,
            "pages": [page.__dict__ for page in self.pages],
        }


def write_prepared_chapter(chapter: PreparedChapter) -> Path:
    out = chapter.root / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chapter.to_manifest(), ensure_ascii=False, indent=2) + "\n", "utf-8")
    return out


def load_prepared_chapter(root: Path) -> PreparedChapter:
    root = Path(root)
    data = json.loads((root / "manifest.json").read_text("utf-8"))
    pages = tuple(PreparedPage(**page) for page in data["pages"])
    return PreparedChapter(
        root=root,
        source=data.get("source", ""),
        pages=pages,
        version=int(data.get("version", 1)),
    )
