"""Scan stage domain types — pure Python, no numpy, no vision imports."""

from __future__ import annotations

from dataclasses import dataclass

from .prepared import Chapter as PreparedChapter


@dataclass(frozen=True)
class Box:
    """Render and erase geometry for one bubble, in prepared-page coordinates."""
    polygon:  list[list[float]]
    fit:      list[int]
    erase:    list[int]
    text:     list[int]


@dataclass(frozen=True)
class Bubble:
    """One accepted text group after detect → group → OCR."""
    idx:         int
    page_index:  int
    source_text: str
    confidence:  float
    box:         Box
    shape_kind:  str = "dialogue"   # dialogue | burst


@dataclass(frozen=True)
class BubbleKey:
    """Stable pairing of an opaque translation key with its bubble."""
    key:    str
    bubble: Bubble

    @property
    def page_index(self) -> int:  return self.bubble.page_index
    @property
    def idx(self) -> int:         return self.bubble.idx
    @property
    def source_text(self) -> str: return self.bubble.source_text
    @property
    def box(self) -> Box:         return self.bubble.box


@dataclass(frozen=True)
class Page:
    index:   int
    width:   int
    height:  int
    bubbles: tuple[Bubble, ...]


@dataclass(frozen=True)
class Chapter:
    """Full scan output — typed boundary between scan and translate stages."""
    prepared: PreparedChapter
    pages:    tuple[Page, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def all_bubbles(self) -> list[Bubble]:
        return [b for p in self.pages for b in p.bubbles]


# ── Geometry — serialized to scan.npz, not part of domain chain ──────


@dataclass(frozen=True)
class BubbleGeometry:
    """Polygon + render boxes for one bubble — stored in scan.npz."""
    bubble_idx: int
    polygon:    list[list[float]]
    fit_box:    list[int]
    erase_box:  list[int]
    text_box:   list[int]


@dataclass(frozen=True)
class PageGeometry:
    """All geometry for one page — stored in scan.npz."""
    page_index: int
    width:      int
    height:     int
    bubbles:    tuple[BubbleGeometry, ...]
