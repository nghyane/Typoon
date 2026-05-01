"""Scan stage output types — pure Python, no numpy, no vision imports."""

from __future__ import annotations

from dataclasses import dataclass

from .prepared import Chapter as _PreparedChapter


@dataclass(frozen=True)
class Box:
    """Render and erase geometry for one bubble, in prepared-page coordinates."""

    polygon:  list[list[float]]  # render boundary [[x,y], ...]
    fit:      list[int]          # [x1,y1,x2,y2] tight fit around text
    erase:    list[int]          # [x1,y1,x2,y2] dilated for inpainting
    text:     list[int]          # [x1,y1,x2,y2] raw detection bbox


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


