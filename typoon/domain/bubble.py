"""Domain types — bubble, page, session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..vision.types import TextMask

if TYPE_CHECKING:
    from ..app.events import EventSink
    from ..llm.ir import Provider
    from ..ports import Store


@dataclass
class Bubble:
    """A text bubble — fields filled progressively by each pipeline stage."""

    idx: int
    page_index: int
    polygon: list[list[float]]
    erase_masks: list[TextMask] = field(default_factory=list, repr=False)
    text_masks: list[TextMask] = field(default_factory=list, repr=False)
    source_text: str = ""
    ocr_confidence: float = 1.0
    translated_text: str | None = None
    translation_key: str | None = None
    translation_status: str = "ok"
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
    hook: object  # EventSink or legacy Hook
    chapter: float = 0.0
    glossary: dict[str, str] = field(default_factory=dict)
