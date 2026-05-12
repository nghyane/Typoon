"""TranslateCtx — immutable context for one draft translation run.

Keyed by (chapter_id, draft_id) — the draft owns the lang pair + glossary
fingerprint, the chapter owns pixel-derived state (bubbles, geometry).
project_id is gone; nothing in the translate pipeline needs it now.
"""

from __future__ import annotations

from dataclasses import dataclass

from typoon.llm.ir import Provider
from typoon.runs.events import Hook
from typoon.storage import Store


@dataclass(frozen=True)
class TranslateCtx:
    translation_provider: Provider
    vision_provider:      Provider   # used by both storyboard context pass + look_at
    store:                Store
    chapter_id:           int       # DB primary key
    draft_id:             int       # which draft this run is filling in
    chapter_position:     int       # sort cursor for "before this chapter" lookups
    material_id:          int       # for community_glossary lookups
    owner_id:             int       # draft creator; drives glossary lookup
    source_lang:          str
    target_lang:          str
    hook:                 Hook


def make_ctx(
    chapter_id:       int,
    draft_id:         int,
    chapter_position: int,
    material_id:      int,
    owner_id:         int,
    source_lang:      str,
    target_lang:      str,
    store: Store,
    *,
    config=None,
    hook: Hook | None = None,
) -> TranslateCtx:
    from typoon.config import load_config
    from typoon.providers import (
        make_translation_provider, make_vision_provider,
    )
    from typoon.runs.events import Hook as _Hook

    if config is None:
        config, _ = load_config()
    if hook is None:
        hook = _Hook()

    return TranslateCtx(
        translation_provider=make_translation_provider(config),
        vision_provider=make_vision_provider(config),
        store=store,
        chapter_id=chapter_id,
        draft_id=draft_id,
        chapter_position=chapter_position,
        material_id=material_id,
        owner_id=owner_id,
        source_lang=source_lang,
        target_lang=target_lang,
        hook=hook,
    )
