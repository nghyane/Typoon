"""Scan stage domain types — pure Python, no numpy, no vision imports."""

from __future__ import annotations

from dataclasses import dataclass

from .prepared import Chapter as PreparedChapter


@dataclass(frozen=True)
class Bubble:
    """One accepted text group after detect → group → OCR.

    ``polygon`` is the rendered drawable area in prepared-page
    coordinates — already padded by the grouper; the renderer treats it
    as the fit polygon directly. The erase mask is stored separately in
    the MaskStore alongside the page (pixel-level, not derivable from
    geometry).
    """
    idx:                  int
    page_index:           int
    source_text:          str
    confidence:           float
    polygon:              list[list[float]]
    shape_kind:           str = "dialogue"
    rotation_deg:         float = 0.0
    src_font_size_px:     int = 0
    src_line_count:       int = 0
    src_avg_chars_per_line: float = 0.0
    text_direction:       str = "horizontal"


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
    def polygon(self) -> list[list[float]]: return self.bubble.polygon


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


@dataclass(frozen=True)
class BubbleGeometry:
    """Polygon for one bubble — stored in scan.npz / DB."""
    bubble_idx:             int
    polygon:                list[list[float]]
    rotation_deg:           float = 0.0
    src_font_size_px:       int   = 0
    src_line_count:         int   = 0
    src_avg_chars_per_line: float = 0.0
    text_direction:         str   = "horizontal"


@dataclass(frozen=True)
class PageGeometry:
    """All geometry for one page — stored in scan.npz / DB."""
    page_index: int
    width:      int
    height:     int
    bubbles:    tuple[BubbleGeometry, ...]
