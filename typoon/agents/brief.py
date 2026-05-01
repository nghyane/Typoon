"""ChapterBrief model and text formatting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typoon.domain.scan import Bubble as ScannedBubble


@dataclass(slots=True)
class AddressRule:
    """Xưng hô binding for a speaker→listener pair."""
    speaker: str       # character name/role, e.g. "elf", "Saran", "narrator"
    listener: str      # character name/role, or "*" for general
    self_ref: str      # how speaker refers to self, e.g. "tôi", "ta", "em"
    other_ref: str     # how speaker refers to listener, e.g. "cô", "anh", "cậu"
    note: str = ""     # tone/register note, e.g. "formal", "hostile"

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "listener": self.listener,
            "self_ref": self.self_ref,
            "other_ref": self.other_ref,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AddressRule":
        return cls(
            speaker=d["speaker"],
            listener=d["listener"],
            self_ref=d["self_ref"],
            other_ref=d["other_ref"],
            note=d.get("note", ""),
        )


@dataclass(slots=True)
class ChapterBrief:
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)
    address: list[AddressRule] = field(default_factory=list)  # binding xưng hô
    style_notes: list[str] = field(default_factory=list)      # tone/style hints
    page_notes: dict[int, str] = field(default_factory=dict)
    key_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "facts": self.facts,
            "glossary": self.glossary,
            "address": [r.to_dict() for r in self.address],
            "style_notes": self.style_notes,
            "page_notes": {str(k): v for k, v in self.page_notes.items()},
            "key_notes": self.key_notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterBrief":
        return cls(
            summary=d.get("summary", ""),
            facts=d.get("facts", []),
            glossary=d.get("glossary", {}),
            address=[AddressRule.from_dict(r) for r in d.get("address", [])],
            style_notes=d.get("style_notes", d.get("rules", [])),  # migrate old rules
            page_notes={int(k): v for k, v in d.get("page_notes", {}).items()},
            key_notes=d.get("key_notes", {}),
        )


# ── Text helpers ──────────────────────────────────────────────────────


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
        parts.append("Glossary:\n" + "\n".join(f"- {k} → {v}" for k, v in brief.glossary.items()))

    if brief.address:
        lines = []
        for r in brief.address:
            line = f"- {r.speaker} → {r.listener}: self={r.self_ref}, other={r.other_ref}"
            if r.note:
                line += f" ({r.note})"
            lines.append(line)
        parts.append(
            "Address rules (BINDING — do not deviate):\n" + "\n".join(lines)
        )

    if brief.style_notes:
        parts.append("Style:\n" + "\n".join(f"- {n}" for n in brief.style_notes))

    notes = [brief.page_notes[p] for p in sorted(page_indices) if p in brief.page_notes]
    if notes:
        parts.append("Page notes:\n" + "\n".join(f"- {n}" for n in notes))

    key_notes = [f"#{k}: {brief.key_notes[k]}" for k in keys if k in brief.key_notes]
    if key_notes:
        parts.append("Key notes:\n" + "\n".join(key_notes))

    return "\n\n".join(parts) if parts else "(none)"
