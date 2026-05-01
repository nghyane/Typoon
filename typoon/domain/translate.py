"""Translate stage output types — pure Python, no LLM or adapter imports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scan import Box, Bubble as ScannedBubble, Chapter as ScannedChapter, Page as ScannedPage


@dataclass(frozen=True)
class Bubble:
    """One bubble after translation."""

    source:           ScannedBubble
    translation_key:  str
    translated_text:  str
    kind:             str   # "dialogue" | "sfx" | "skip"

    @property
    def idx(self) -> int:
        return self.source.idx

    @property
    def page_index(self) -> int:
        return self.source.page_index

    @property
    def source_text(self) -> str:
        return self.source.source_text


@dataclass(frozen=True)
class Page:
    """One page after translation."""

    source:  ScannedPage
    bubbles: tuple[Bubble, ...]

    @property
    def index(self) -> int:
        return self.source.index


@dataclass(frozen=True)
class Chapter:
    """Full translate output — typed boundary between translate and render stages."""

    scan:  ScannedChapter
    pages: tuple[Page, ...]

    @property
    def all_bubbles(self) -> list[Bubble]:
        return [b for p in self.pages for b in p.bubbles]

    def save(self, cp) -> Path:
        """Save to cp.translate. cp is a ChapterPaths instance."""
        data: dict[str, Any] = {
            "version": 1,
            "bubbles": [
                {
                    "idx":             b.idx,
                    "page_index":      b.page_index,
                    "translation_key": b.translation_key,
                    "source_text":     b.source_text,
                    "translated_text": b.translated_text,
                    "kind":            b.kind,
                    "confidence":      b.source.confidence,
                    "box": {
                        "polygon": b.source.box.polygon,
                        "fit":     b.source.box.fit,
                        "erase":   b.source.box.erase,
                        "text":    b.source.box.text,
                    },
                }
                for b in self.all_bubbles
            ],
        }
        cp.translate.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        return cp.translate

    @classmethod
    def load(cls, cp) -> "Chapter":
        """Load from cp.translate + cp.scan. cp is a ChapterPaths instance."""
        scan = ScannedChapter.load(cp)
        data = json.loads(cp.translate.read_text("utf-8"))

        scan_bubble: dict[tuple[int, int], ScannedBubble] = {
            (b.page_index, b.idx): b for b in scan.all_bubbles
        }

        by_page: dict[int, list[Bubble]] = {}
        for bd in data["bubbles"]:
            key = (bd["page_index"], bd["idx"])
            sb = scan_bubble.get(key) or ScannedBubble(
                idx=bd["idx"],
                page_index=bd["page_index"],
                source_text=bd["source_text"],
                confidence=bd["confidence"],
                box=Box(
                    polygon=bd["box"]["polygon"],
                    fit=bd["box"]["fit"],
                    erase=bd["box"]["erase"],
                    text=bd["box"]["text"],
                ),
            )
            by_page.setdefault(bd["page_index"], []).append(Bubble(
                source=sb,
                translation_key=bd["translation_key"],
                translated_text=bd["translated_text"],
                kind=bd["kind"],
            ))

        pages = tuple(
            Page(source=sp, bubbles=tuple(by_page.get(sp.index, [])))
            for sp in scan.pages
        )
        return cls(scan=scan, pages=pages)
