"""Prompt templates and policy loading.

Templates live as markdown files under `prompts/`. We load lazily —
the file read happens on first `.format()` call, not at module import
— so a missing prompt file doesn't crash the import graph and tests
that don't exercise the LLM path don't pay the I/O.

`Template` is a small wrapper rather than `str` subclass: subclassing
`str` to override `format` confuses type checkers and any code that
expects `str` semantics (concatenation, slicing, etc.). The wrapper
exposes `.format()` and `__str__()` and nothing else.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Map ISO codes to full English names. Full names are what the prompt
# shows to the model — code-style `ja`/`vi` is ambiguous to small
# models and conflicts with policy files that already say "Japanese"
# / "Vietnamese". Unknown codes pass through unchanged so we don't
# silently mistranslate an unfamiliar locale tag.
_LANG_NAMES = {
    "en": "English",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}


def lang_name(code: str) -> str:
    return _LANG_NAMES.get(code, code)


@lru_cache(maxsize=64)
def load_policy(name: str) -> str:
    """Read a prompt file relative to `prompts/`. Cached so repeated
    calls during a long-running worker stay free after warmup."""
    path = _PROMPTS_DIR / name
    if not path.exists():
        logger.warning("prompt file not found: %s", path)
        return ""
    return path.read_text("utf-8").strip()


def load_source_policy(source_lang: str) -> str:
    return load_policy(f"sources/{source_lang}.md")


def load_target_policy(target_lang: str) -> str:
    return load_policy(f"targets/{target_lang}.md")


class Template:
    """Lazy prompt template. `.format(**kwargs)` reads the file the
    first time and reuses the cached content thereafter (per the
    `load_policy` LRU)."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def format(self, **kwargs: object) -> str:
        return load_policy(self._name).format(**kwargs)

    def __repr__(self) -> str:
        return f"Template({self._name!r})"


STORYBOARD_SYSTEM = Template("agents/storyboard.md")
PAGE_SYSTEM       = Template("agents/page.md")
