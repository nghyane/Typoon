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


def make_session(
    project_id: int,
    chapter: float,
    source_lang: str,
    target_lang: str,
    store: object,
    *,
    config=None,
) -> Session:
    """Build a Session from config. Loads providers from config.toml."""
    from typoon.config import load_config
    from typoon.providers import make_context_provider, make_translation_provider
    from typoon.runs.events import Hook

    if config is None:
        config, _ = load_config()

    return Session(
        store=store,
        source=None,
        project_id=project_id,
        source_lang=source_lang,
        target_lang=target_lang,
        provider=make_translation_provider(config),
        context_provider=make_context_provider(config),
        hook=Hook(),
        chapter=chapter,
        glossary={},
    )
