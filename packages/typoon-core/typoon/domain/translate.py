"""Translate stage domain types — pure Python, no LLM or adapter imports."""

from __future__ import annotations

from dataclasses import dataclass

from .scan import Bubble as ScannedBubble, Chapter as ScannedChapter, Page as ScannedPage


@dataclass(frozen=True)
class Bubble:
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

    def to_db_records(self) -> list[dict]:
        """Return flat list for storage.save_translations().
        Identity: (page_index, bubble_idx) — structural, not hash-based.
        """
        return [
            {
                "page_index":      b.page_index,
                "bubble_idx":      b.idx,
                "translated_text": b.translated_text,
                "kind":            b.kind,
            }
            for b in self.all_bubbles
            if b.kind != "skip"
        ]
