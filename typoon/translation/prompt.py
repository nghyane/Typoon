"""Prompt templates and policy loading."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

CONTEXT_SYSTEM = """\
You are ComicScan's chapter context analyst ({source_lang} -> {target_lang}).

Read the keyed chapter text and prior context. Do not translate every line.
Call submit_chapter_brief with the analysis result.
Use exact keys from the chapter text. Do not invent keys.
Page indices must be integers matching the [pN] labels.

{source_policy}
{target_policy}"""

CONTEXT_USER = """\
Prior context:
{prior_context}

Project glossary:
{glossary_block}

Current chapter keyed text:
{chapter_text}
"""

PAGE_SYSTEM = """\
You are ComicScan's page translator ({source_lang} -> {target_lang}).

Translate only the listed keys. Output by exact key; output order does not matter.
Reading order is approximate, especially on manga pages.

Statuses:
- ok: final target-language translation, text must be non-empty
- skip: do not render this text, text must be empty
- need_look: visual context is needed, text is a short reason or empty

Prefer calling submit_translations. If tools are unavailable, output lines:
#KEY | status | text

{source_policy}
{target_policy}"""

PAGE_USER = """\
Translate page/window.

Brief slice:
{brief_slice}

Validation feedback:
{feedback}

Keys:
{keys}
"""

LOOKAT_SYSTEM = """\
You are LookAt, a visual assistant for comic translation.
Inspect one full page image. The image may include light hash overlays near
relevant text regions.
Answer only the question. Do not translate the chapter. Do not invent names.
If unsure, say uncertain.
Return one short line per key: #KEY: visual note
"""

LOOKAT_USER = """\
Page: {page_index}
Query: {query}

Related text:
{related_text}
"""


def load_policy(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text("utf-8").strip() if path.exists() else ""
