"""Render stage output types — carries pixel data, not a pure domain type.

RenderedPage holds the final composited image alongside its source metadata.
Pixel data lives here intentionally — render is the last stage and this is
the output artifact consumed by file writing / UI display.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .translate import TranslatedBubble, TranslatedChapter, TranslatedPage


@dataclass(frozen=True)
class RenderedBubble:
    """One bubble after text layout — geometry and render metadata only."""

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
class RenderedPage:
    """One page after full render — pixel image + rendered bubble metadata."""

    source:  TranslatedPage
    bubbles: tuple[RenderedBubble, ...]
    image:   np.ndarray = field(repr=False)   # RGB uint8 (H, W, 3)

    @property
    def index(self) -> int:
        return self.source.index


@dataclass(frozen=True)
class RenderedChapter:
    """Full render output — final pipeline boundary."""

    source: TranslatedChapter
    pages:  tuple[RenderedPage, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)
