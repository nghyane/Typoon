"""Session — carries everything translation agents need for one chapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typoon.llm.ir import Provider


@dataclass
class Session:
    """Created once per chapter. Passed through all translation sub-stages."""

    store: object
    source: object
    project_id: int
    source_lang: str
    target_lang: str
    provider: "Provider"
    context_provider: "Provider"
    hook: object
    chapter: float = 0.0
    glossary: dict[str, str] = field(default_factory=dict)
