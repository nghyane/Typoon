"""PreparedChapter manifest contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Page:
    index:  int
    file:   str
    width:  int
    height: int


@dataclass(frozen=True)
class Chapter:
    root:   Path
    source: str
    pages:  tuple[Page, ...]
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
            "pages": [p.__dict__ for p in self.pages],
        }

    def save(self) -> Path:
        out = self.root / "manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_manifest(), ensure_ascii=False, indent=2) + "\n", "utf-8")
        return out

    @classmethod
    def load(cls, root: Path) -> "Chapter":
        root = Path(root)
        data = json.loads((root / "manifest.json").read_text("utf-8"))
        pages = tuple(Page(**p) for p in data["pages"])
        return cls(
            root=root,
            source=data.get("source", ""),
            pages=pages,
            version=int(data.get("version", 1)),
        )


