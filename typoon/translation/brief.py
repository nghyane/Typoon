"""ChapterBrief model and formatting helpers."""

from __future__ import annotations

import json
import re
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

    @staticmethod
    def from_json(text: str) -> "ChapterBrief":
        data = _extract_json(text)
        return ChapterBrief(
            summary=str(data.get("summary", "")),
            facts=[str(x) for x in data.get("facts", []) or []],
            glossary={str(k): str(v) for k, v in (data.get("glossary", {}) or {}).items()},
            style_rules=[str(x) for x in data.get("style_rules", []) or []],
            pronoun_rules=[str(x) for x in data.get("pronoun_rules", []) or []],
            page_notes={_page_index(k): str(v) for k, v in (data.get("page_notes", {}) or {}).items()},
            key_notes={str(k): str(v) for k, v in (data.get("key_notes", {}) or {}).items()},
            look_requests=[
                LookRequest(
                    page_index=_page_index(x.get("page_index", 0)),
                    keys=[str(k) for k in x.get("keys", [])],
                    query=str(x.get("query", "")),
                )
                for x in data.get("look_requests", []) or []
                if isinstance(x, dict)
            ],
        )


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


def _extract_json(text: str) -> dict:
    s = text.strip()
    if "```" in s:
        chunks = s.split("```")
        for chunk in chunks:
            c = chunk.strip()
            if c.startswith("json"):
                c = c[4:].strip()
            if c.startswith("{"):
                return json.loads(c)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start:end + 1])
    return {}


def _page_index(value: object) -> int:
    """Parse model page labels such as 0, "0", "p0", or "page 0"."""
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else 0
