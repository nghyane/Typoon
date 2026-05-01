"""Render stage output types."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .translate import Bubble as TranslatedBubble, Chapter as TranslatedChapter, Page as TranslatedPage


@dataclass(frozen=True)
class Bubble:
    """One bubble after text layout."""

    source:    TranslatedBubble
    font_size: int
    overflow:  bool

    @property
    def idx(self) -> int:
        return self.source.idx

    @property
    def page_index(self) -> int:
        return self.source.page_index

    @property
    def translated_text(self) -> str:
        return self.source.translated_text

    @property
    def kind(self) -> str:
        return self.source.kind


@dataclass
class Page:
    """One page after full render — pixel image + bubble metadata."""

    source:  TranslatedPage
    bubbles: tuple[Bubble, ...]
    image:   np.ndarray = field(repr=False)  # RGB uint8 (H, W, 3)

    @property
    def index(self) -> int:
        return self.source.index


@dataclass(frozen=True)
class Chapter:
    """Full render output — final pipeline boundary."""

    source: TranslatedChapter
    pages:  tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)


