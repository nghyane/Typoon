"""Scan stage output types — pure Python, no numpy, no vision imports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prepared import Chapter as _PreparedChapter


@dataclass(frozen=True)
class Box:
    """Render and erase geometry for one bubble, in prepared-page coordinates."""

    polygon:  list[list[float]]
    fit:      list[int]
    erase:    list[int]
    text:     list[int]


@dataclass(frozen=True)
class BubbleKey:
    """Stable pairing of an opaque translation key with its bubble.

    Single source of truth for bubble identity throughout the translate stage.
    Use key for LLM communication; use (page_index, idx) for structural lookup.
    """
    key:    str
    bubble: "Bubble"

    @property
    def page_index(self) -> int:  return self.bubble.page_index
    @property
    def idx(self) -> int:         return self.bubble.idx
    @property
    def source_text(self) -> str: return self.bubble.source_text
    @property
    def box(self):                return self.bubble.box


@dataclass(frozen=True)
class Bubble:
    """One accepted text group after detect → group → OCR."""

    idx:         int
    page_index:  int
    source_text: str
    confidence:  float
    box:         Box


@dataclass(frozen=True)
class Page:
    """One prepared page after scan stage."""

    index:   int
    width:   int
    height:  int
    bubbles: tuple[Bubble, ...]


@dataclass(frozen=True)
class Chapter:
    """Full scan output — typed boundary between scan and translate stages."""

    prepared: _PreparedChapter
    pages:    tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_bubbles(self) -> list[Bubble]:
        return [b for p in self.pages for b in p.bubbles]

    def save(self, cp) -> Path:
        """Save to cp.scan (scan.json). cp is a ChapterPaths instance."""
        data: dict[str, Any] = {
            "version": 1,
            "prepared_root": str(self.prepared.root),
            "pages": [
                {
                    "index":   p.index,
                    "width":   p.width,
                    "height":  p.height,
                    "bubbles": [
                        {
                            "idx":         b.idx,
                            "page_index":  b.page_index,
                            "source_text": b.source_text,
                            "confidence":  b.confidence,
                            "box": {
                                "polygon": b.box.polygon,
                                "fit":     b.box.fit,
                                "erase":   b.box.erase,
                                "text":    b.box.text,
                            },
                        }
                        for b in p.bubbles
                    ],
                }
                for p in self.pages
            ],
        }
        cp.scan.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        return cp.scan

    @classmethod
    def load(cls, cp) -> "Chapter":
        """Load from cp.scan. cp is a ChapterPaths instance."""
        from .prepared import Chapter as PreparedChapter
        data = json.loads(cp.scan.read_text("utf-8"))
        prepared = PreparedChapter.load(data["prepared_root"])
        pages = tuple(
            Page(
                index=pd["index"],
                width=pd["width"],
                height=pd["height"],
                bubbles=tuple(
                    Bubble(
                        idx=b["idx"],
                        page_index=b["page_index"],
                        source_text=b["source_text"],
                        confidence=b["confidence"],
                        box=Box(
                            polygon=b["box"]["polygon"],
                            fit=b["box"]["fit"],
                            erase=b["box"]["erase"],
                            text=b["box"]["text"],
                        ),
                    )
                    for b in pd["bubbles"]
                ),
            )
            for pd in data["pages"]
        )
        return cls(prepared=prepared, pages=pages)
