"""Translate stage output types — pure Python, no LLM or adapter imports."""

from __future__ import annotations

from dataclasses import dataclass

from .scan import ScannedBubble, ScannedChapter, ScannedPage


@dataclass(frozen=True)
class TranslatedBubble:
    """One bubble after translation — carries scan output as immutable reference."""

    source:             ScannedBubble
    translation_key:    str
    translated_text:    str
    kind:               str     # "dialogue" | "sfx" | "skip"

    # Convenience pass-throughs so callers don't need to reach into source.
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
class TranslatedPage:
    """One page after translation."""

    source:  ScannedPage
    bubbles: tuple[TranslatedBubble, ...]

    @property
    def index(self) -> int:
        return self.source.index


@dataclass(frozen=True)
class TranslatedChapter:
    """Full translate output — typed boundary between translate and render stages."""

    scan:  ScannedChapter
    pages: tuple[TranslatedPage, ...]

    @property
    def all_bubbles(self) -> list[TranslatedBubble]:
        return [b for p in self.pages for b in p.bubbles]
