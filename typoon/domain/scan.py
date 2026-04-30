"""Scan stage output types — pure Python, no numpy, no vision imports."""

from __future__ import annotations

from dataclasses import dataclass

from .prepared import PreparedChapter


@dataclass(frozen=True)
class BubbleGeometry:
    """Render and erase geometry for one bubble, in prepared-page coordinates."""

    polygon:    list[list[float]]   # render boundary [[x,y], ...]
    fit_bbox:   list[int]           # [x1,y1,x2,y2] tight fit around text
    erase_bbox: list[int]           # [x1,y1,x2,y2] dilated for inpainting
    text_bbox:  list[int]           # [x1,y1,x2,y2] raw detection bbox


@dataclass(frozen=True)
class ScannedBubble:
    """One accepted text group after detect → group → OCR."""

    idx:         int
    page_index:  int
    source_text: str
    confidence:  float
    geometry:    BubbleGeometry


@dataclass(frozen=True)
class ScannedPage:
    """One prepared page after scan stage."""

    index:   int
    width:   int
    height:  int
    bubbles: tuple[ScannedBubble, ...]


@dataclass(frozen=True)
class ScannedChapter:
    """Full scan output — typed boundary between scan and translate stages."""

    prepared: PreparedChapter
    pages:    tuple[ScannedPage, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_bubbles(self) -> list[ScannedBubble]:
        return [b for p in self.pages for b in p.bubbles]
