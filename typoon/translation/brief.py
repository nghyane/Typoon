"""ChapterBrief model and formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LookRequest:
    page_index: int
    keys: list[str]
    query: str


@dataclass(slots=True)
class ChapterBrief:
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)
    style_rules: list[str] = field(default_factory=list)
    pronoun_rules: list[str] = field(default_factory=list)
    page_notes: dict[int, str] = field(default_factory=dict)
    key_notes: dict[str, str] = field(default_factory=dict)
    look_requests: list[LookRequest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "facts": self.facts,
            "glossary": self.glossary,
            "style_rules": self.style_rules,
            "pronoun_rules": self.pronoun_rules,
            "page_notes": {str(k): v for k, v in self.page_notes.items()},
            "key_notes": self.key_notes,
            "look_requests": [r.__dict__ for r in self.look_requests],
        }


def chapter_text(pages: list) -> str:
    lines = []
    for page in pages:
        for b in page.bubbles:
            key = b.translation_key
            if key is None:
                raise ValueError(f"Bubble {b.id} has no translation key")
            lines.append(f"[p{b.page_index}] #{key} {b.source_text}")
    return "\n".join(lines)


def brief_slice(brief: ChapterBrief, page_indices: set[int], keys: list[str]) -> str:
    parts: list[str] = []
    if brief.summary:
        parts.append(f"Summary: {brief.summary}")
    if brief.glossary:
        parts.append("Glossary:\n" + "\n".join(f"- {k} => {v}" for k, v in brief.glossary.items()))
    rules = brief.style_rules + brief.pronoun_rules
    if rules:
        parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
    notes = [brief.page_notes[p] for p in sorted(page_indices) if p in brief.page_notes]
    if notes:
        parts.append("Page notes:\n" + "\n".join(f"- {n}" for n in notes))
    key_notes = [f"#{k}: {brief.key_notes[k]}" for k in keys if k in brief.key_notes]
    if key_notes:
        parts.append("Key notes:\n" + "\n".join(key_notes))
    return "\n\n".join(parts) if parts else "(none)"
