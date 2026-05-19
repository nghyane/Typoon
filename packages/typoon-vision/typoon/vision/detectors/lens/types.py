"""Internal dataclasses shared across the Lens detector package."""

from __future__ import annotations

from dataclasses import dataclass

from ...contracts import LineBox, WordBox


@dataclass(slots=True)
class RawBlock:
    """Tile-local intermediate before page-coordinate translation.

    Carries everything we need from a single Lens paragraph: page-pixel
    bbox, recognised text, rotation, per-word + per-line geometry, and
    the writing direction projected onto page-aligned axes.
    """
    bbox:           tuple[int, int, int, int]
    text:           str
    confidence:     float
    rotation_deg:   float
    words:          tuple[WordBox, ...]
    lines:          tuple[LineBox, ...]
    text_direction: str        # "horizontal" | "vertical" on page axes
