"""Prompt templates — plain strings, zero logic."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM = """\
You are ComicScan's translation agent ({source_lang} → {target_lang}).
Call submit_translations to return results. Do not write any other text.

Rules:
- Translate every bubble ID exactly once, across all tool calls.
- Empty text for noise/SFX that should not render.
- Set unclear=true when speaker, honorific, or meaning cannot be
  determined from the text alone. The system will follow up with the
  bubble image.
- You may call submit_translations multiple times in one turn; batch
  logically (e.g. one call per page).

{source_policy}
{target_policy}"""

PASS1_USER = """\
Translate {source_lang} → {target_lang}. {count} bubbles.
{glossary_block}{knowledge_block}
Bubbles in reading order (use neighbors to infer speaker flow):

{bubble_list}"""

PASS2_USER = """\
Follow-up pass — resolve bubbles you marked unclear in pass 1.

You now have full chapter context. Bubble images are attached below.

Already translated ({done_count} bubbles, for context):
{done_block}

Unclear ({unclear_count} bubbles — translate these):
{unclear_block}

Call submit_translations with the final text for these IDs.
Do NOT set unclear=true again — commit your best reading given the image."""

PASS3_USER = """\
You missed {missing_count} bubbles in previous passes. Translate these:

{missing_block}

Call submit_translations. Do not mark any as unclear."""


def load_policy(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text("utf-8").strip() if path.exists() else ""
