"""Prompt templates and policy loading."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_policy(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.exists():
        logger.warning("prompt file not found: %s", path)
        return ""
    return path.read_text("utf-8").strip()


def load_source_policy(source_lang: str) -> str:
    return load_policy(f"sources/{source_lang}.md")


def load_target_policy(target_lang: str) -> str:
    return load_policy(f"targets/{target_lang}.md")


def _lazy(name: str) -> str:
    """Load on first use — not at import time."""
    val = load_policy(name)
    return val


class _LazyTemplate(str):
    """String subclass that loads its content on first format() call."""
    def __new__(cls, name: str) -> "_LazyTemplate":
        obj = str.__new__(cls, "")
        obj._name = name  # type: ignore[attr-defined]
        obj._loaded: str | None = None  # type: ignore[attr-defined]
        return obj

    def format(self, **kwargs) -> str:  # type: ignore[override]
        if self._loaded is None:
            self._loaded = load_policy(self._name)
        return self._loaded.format(**kwargs)


STORYBOARD_SYSTEM = _LazyTemplate("agents/storyboard.md")
PAGE_SYSTEM       = _LazyTemplate("agents/page.md")
