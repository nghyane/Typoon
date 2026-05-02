"""Prompt templates and policy loading."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_policy(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text("utf-8").strip() if path.exists() else ""


CONTEXT_SYSTEM = load_policy("agents/context.md")

CONTEXT_USER = """\
{context_snapshot}

## Chapter text to analyze
{chapter_text}"""

PAGE_SYSTEM = load_policy("agents/page.md")


def load_source_policy(source_lang: str) -> str:
    return load_policy(f"sources/{source_lang}.md") or load_policy(f"source_{source_lang}.md")


def load_target_policy(target_lang: str) -> str:
    return load_policy(f"targets/{target_lang}.md") or load_policy(f"target_{target_lang}.md")
