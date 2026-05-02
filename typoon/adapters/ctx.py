"""TranslateCtx — immutable context for one chapter translation run."""

from __future__ import annotations

from dataclasses import dataclass

from typoon.storage.store import Store
from typoon.llm.ir import Provider
from typoon.runs.events import Hook


@dataclass(frozen=True)
class TranslateCtx:
    translation_provider: Provider
    context_provider:     Provider
    vision_provider:      Provider
    store:                Store
    project_id:           int
    chapter_id:           int       # DB primary key
    chapter_idx:          float     # chapter number (for brief lookup by idx)
    source_lang:          str
    target_lang:          str
    hook:                 Hook


def make_ctx(
    project_id: int,
    chapter_id: int,
    chapter_idx: float,
    source_lang: str,
    target_lang: str,
    store: Store,
    *,
    config=None,
    hook: Hook | None = None,
) -> TranslateCtx:
    from typoon.config import load_config
    from typoon.providers import make_context_provider, make_translation_provider, make_vision_provider
    from typoon.runs.events import Hook as _Hook

    if config is None:
        config, _ = load_config()
    if hook is None:
        hook = _Hook()

    return TranslateCtx(
        translation_provider=make_translation_provider(config),
        context_provider=make_context_provider(config),
        vision_provider=make_vision_provider(config),
        store=store,
        project_id=project_id,
        chapter_id=chapter_id,
        chapter_idx=chapter_idx,
        source_lang=source_lang,
        target_lang=target_lang,
        hook=hook,
    )
