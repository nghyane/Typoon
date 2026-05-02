"""Render stage domain types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter, Page as TranslatedPage


@dataclass(frozen=True)
class Bubble:
    source:    TranslatedBubble
    font_size: int
    overflow:  bool

    @property
    def idx(self) -> int:             return self.source.idx
    @property
    def page_index(self) -> int:      return self.source.page_index
    @property
    def translated_text(self) -> str: return self.source.translated_text
    @property
    def kind(self) -> str:            return self.source.kind


@dataclass(frozen=True)
class Page:
    source:     TranslatedPage
    bubbles:    tuple[Bubble, ...]
    image_path: Path | None = None

    @property
    def index(self) -> int:
        return self.source.index


@dataclass(frozen=True)
class Chapter:
    source: TranslatedChapter
    pages:  tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)
