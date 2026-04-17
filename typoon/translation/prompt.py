"""Prompt templates — plain strings, zero logic."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SYSTEM = """\
You are ComicScan's translation agent ({source_lang} → {target_lang}).
Respond ONLY with tool calls. No explanatory text.
{source_policy}
{target_policy}"""

USER = """\
Translate {source_lang} → {target_lang}. {count} bubbles.
Required IDs: {ids}
{glossary_block}
{knowledge_block}
Bubbles in reading order (use neighbors to infer speaker flow):

{bubble_list}"""


def load_policy(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text("utf-8").strip() if path.exists() else ""
