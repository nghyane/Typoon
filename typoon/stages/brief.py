"""ChapterBrief — translator-facing context for one chapter.

Produced by `stages.scan_context.build_chapter_context` via a single vision
call per storyboard, no multi-turn agent. Drop-in replacement for the old
agent-emitted ChapterBrief; shape is intentionally smaller — only fields
the translator (page.py) or the render path actually consume.

What the translator reads (via `brief_slice`):
  - glossary  → name mappings discovered by vision
  - style_notes → register/mood/translator-guidance lines
  - key_notes → per-bubble speaker hint

What the render path reads:
  - noise_keys / noise_pages → skip rendering

What material memory persists (not consumed by translator directly):
  - characters → full discovered character data (name, gender, role)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typoon.domain.scan import BubbleKey


@dataclass(slots=True)
class Character:
    """One character discovered by the vision pass.

    `gender` is one of "male" | "female" | "unknown". `role` is a short
    free-text descriptor ("young man", "elderly devil hunter", etc.) —
    helpful for the translator + material memory, not enforced.
    """
    name:   str
    gender: str = "unknown"
    role:   str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "gender": self.gender, "role": self.role}

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        return cls(
            name=d["name"],
            gender=d.get("gender", "unknown"),
            role=d.get("role", ""),
        )


@dataclass(slots=True)
class ChapterBrief:
    glossary:    dict[str, str]      = field(default_factory=dict)
    style_notes: list[str]           = field(default_factory=list)
    key_notes:   dict[str, str]      = field(default_factory=dict)
    characters:  list[Character]     = field(default_factory=list)
    noise_keys:  set[str]            = field(default_factory=set)
    # Page indices that are entirely non-diegetic (full-page credits,
    # ads, platform banners). These pages are dropped from the public
    # render archive entirely.
    noise_pages: set[int]            = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "glossary":    self.glossary,
            "style_notes": self.style_notes,
            "key_notes":   self.key_notes,
            "characters":  [c.to_dict() for c in self.characters],
            "noise_keys":  sorted(self.noise_keys),
            "noise_pages": sorted(self.noise_pages),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterBrief":
        return cls(
            glossary=dict(d.get("glossary", {})),
            style_notes=list(d.get("style_notes", [])),
            key_notes=dict(d.get("key_notes", {})),
            characters=[Character.from_dict(c) for c in d.get("characters", [])],
            noise_keys=set(d.get("noise_keys", [])),
            noise_pages={int(p) for p in d.get("noise_pages", [])},
        )

    def summary_line(self) -> str | None:
        """Single-sentence summary for memory_briefs.summary indexing.

        Derived deterministically from style_notes; we no longer ask the
        agent for a free-form summary because the vision pass focuses on
        per-bubble structure, not plot recap.
        """
        for note in self.style_notes:
            n = note.strip()
            if n:
                return n[:200]
        return None


def brief_slice(brief: ChapterBrief, page_indices: set[int], keys: list[str]) -> str:
    """Render the subset of brief data relevant to one translation window.

    `page_indices` is unused after the page_notes field was dropped — kept
    in the signature so the caller (page.py) does not need to change shape
    if we re-introduce per-page context later.
    """
    _ = page_indices  # reserved
    parts: list[str] = []

    if brief.glossary:
        parts.append(
            "Glossary:\n" + "\n".join(f"- {k} → {v}" for k, v in brief.glossary.items())
        )

    if brief.characters:
        lines = []
        for c in brief.characters:
            gender = c.gender if c.gender and c.gender != "unknown" else "?"
            role = c.role if c.role else ""
            line = f"- {c.name} ({gender})"
            if role:
                line += f": {role}"
            lines.append(line)
        parts.append("Characters:\n" + "\n".join(lines))

    if brief.style_notes:
        parts.append("Style:\n" + "\n".join(f"- {n}" for n in brief.style_notes))

    key_notes = [
        f"#{k}: {brief.key_notes[k]}"
        for k in keys if k in brief.key_notes
    ]
    if key_notes:
        parts.append("Speaker hints:\n" + "\n".join(key_notes))

    return "\n\n".join(parts) if parts else "(none)"
