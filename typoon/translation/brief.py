"""ChapterBrief model and text formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typoon.domain.scan import ScannedBubble


@dataclass(slots=True)
class ChapterBrief:
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    page_notes: dict[int, str] = field(default_factory=dict)
    key_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "facts": self.facts,
            "glossary": self.glossary,
            "rules": self.rules,
            "page_notes": {str(k): v for k, v in self.page_notes.items()},
            "key_notes": self.key_notes,
        }


def _sorted_items(
    key_map: dict[str, ScannedBubble],
) -> list[tuple[str, ScannedBubble]]:
    return sorted(key_map.items(), key=lambda kv: (kv[1].page_index, kv[1].idx))


def chapter_text(key_map: dict[str, ScannedBubble]) -> str:
    lines = [
        f"[p{b.page_index}] #{key} {b.source_text}"
        for key, b in _sorted_items(key_map)
    ]
    return "\n".join(lines)


def annotated_chapter_text(
    key_map: dict[str, ScannedBubble],
    active_keys: set[str],
) -> str:
    lines = []
    for key, b in _sorted_items(key_map):
        prefix = ">>> " if key in active_keys else "    "
        lines.append(f"{prefix}[p{b.page_index}] #{key} {b.source_text}")
    return "\n".join(lines)


def brief_slice(brief: ChapterBrief, page_indices: set[int], keys: list[str]) -> str:
    parts: list[str] = []
    if brief.summary:
        parts.append(f"Summary: {brief.summary}")
    if brief.glossary:
        parts.append("Glossary:\n" + "\n".join(f"- {k} => {v}" for k, v in brief.glossary.items()))
    if brief.rules:
        parts.append("Rules:\n" + "\n".join(f"- {r}" for r in brief.rules))
    notes = [brief.page_notes[p] for p in sorted(page_indices) if p in brief.page_notes]
    if notes:
        parts.append("Page notes:\n" + "\n".join(f"- {n}" for n in notes))
    key_notes = [f"#{k}: {brief.key_notes[k]}" for k in keys if k in brief.key_notes]
    if key_notes:
        parts.append("Key notes:\n" + "\n".join(key_notes))
    return "\n\n".join(parts) if parts else "(none)"
