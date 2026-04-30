"""Legacy bridge types — Bubble and Page.

These are used only by translation agents (context.py, page.py) while
they are being migrated to work with ScannedBubble/TranslatedBubble.
New code must not add fields here. Pixel data (masks, erased, rendered)
belongs in adapters/mask_store.py and domain/render.py respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Bubble:
    """Translation agent bridge — scan identity + translation fields only."""

    idx:                int
    page_index:         int
    polygon:            list[list[float]]
    source_text:        str = ""
    ocr_confidence:     float = 1.0
    translated_text:    str | None = None
    translation_key:    str | None = None
    translation_status: str = "dialogue"

    @property
    def id(self) -> str:
        return f"p{self.page_index}_b{self.idx}"


@dataclass
class Page:
    """Translation agent bridge — index + bubbles only."""

    index:   int
    bubbles: list[Bubble] = field(default_factory=list)
