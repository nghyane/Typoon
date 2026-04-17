"""Shared value types — no domain imports, no protocols.

Types at the bottom of the dependency graph:
  types.py → (vision.types only)
  ports.py → types.py
  everything else → types.py + ports.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .vision.types import TextMask

if TYPE_CHECKING:
    from .events import Hook
    from .llm.ir import Provider
    from .ports import Store


# ── Pipeline types ───────────────────────────────────────────────


@dataclass
class Bubble:
    """A text bubble — fields filled progressively by each pipeline stage."""

    idx: int
    page_index: int
    polygon: list[list[float]]
    masks: list[TextMask] = field(default_factory=list, repr=False)
    source_text: str = ""
    ocr_confidence: float = 1.0
    translated_text: str | None = None
    font_size: int = 0
    overflow: bool = False

    @property
    def id(self) -> str:
        return f"p{self.page_index}_b{self.idx}"


@dataclass
class Page:
    """A comic page — accumulates results through the pipeline."""

    index: int
    bubbles: list[Bubble] = field(default_factory=list)
    erased: np.ndarray | None = field(default=None, repr=False)
    rendered: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Session:
    """Carries everything agents need. Created once per chapter."""

    store: Store
    source: object
    project_id: int
    source_lang: str
    target_lang: str
    provider: Provider
    context_provider: Provider
    hook: Hook
    glossary: dict[str, str] = field(default_factory=dict)
    knowledge: str | None = None


# ── Connector types ──────────────────────────────────────────────


@dataclass(slots=True)
class ChapterVariant:
    """One upload of a chapter (scanlation group)."""

    id: str
    url: str
    group: str | None = None
    votes: int = 0


@dataclass(slots=True)
class DiscoveredChapter:
    """A chapter discovered from a remote source."""

    number: float
    title: str | None = None
    variants: list[ChapterVariant] = field(default_factory=list)

    @property
    def best_variant(self) -> ChapterVariant:
        return max(self.variants, key=lambda v: v.votes) if self.variants else self.variants[0]


@dataclass(slots=True)
class SourceInfo:
    """Metadata discovered from a manga source URL."""

    suggested_title: str
    suggested_lang: str = "ko"
    cover_url: str | None = None
    chapters: list[DiscoveredChapter] = field(default_factory=list)
