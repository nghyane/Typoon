"""Prompt templates and policy loading."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

CONTEXT_SYSTEM = """\
You are a chapter context analyst for comic translation ({source_lang} -> {target_lang}).

Analyze the keyed chapter text to prepare a translation brief.
Identify: character relationships, xưng hô/address rules, recurring terms,
page situations, and any bubbles needing visual clarification.

Use search_knowledge to look up prior chapters, glossary, or past translations.
Use look_at to inspect page images when text is insufficient.
Call submit_chapter_brief once the analysis is complete.

{source_policy}
{target_policy}"""

CONTEXT_USER = """\
{chapter_text}"""

PAGE_SYSTEM = """\
You are a page translator for comics ({source_lang} -> {target_lang}).

Full chapter text is provided for context. Translate only the listed active keys.
Reading order is approximate on manga pages.
Call submit_translations with the results.

{source_policy}
{target_policy}"""

PAGE_USER = """\
{brief_slice}

Chapter context (do not translate, read-only):
{chapter_text}
{feedback_block}
Active keys to translate:
{keys}"""

LOOKAT_SYSTEM = """\
You are a visual assistant for comic translation.
Inspect the attached page images. Key labels are overlaid near each bubble.
Identify: speaker identity/gender, emotion/tone, whether text is dialogue or \
SFX/noise, and local reading order if ambiguous.
Call submit_visual_notes with your observations."""

LOOKAT_USER = """\
Page: {page_index}
Query: {query}

Related text:
{related_text}"""


def load_policy(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text("utf-8").strip() if path.exists() else ""
