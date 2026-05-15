"""ChapterBrief — translator-facing context for one chapter.

Pure data contract. Produced by `stages.brief.build_chapter_brief` from a
single vision pass over storyboard chunks. Consumed by translate (per-bubble
hints, glossary, style) and render (noise filtering).

Belongs in `domain/` because it crosses stage boundaries (stages.brief
produces, stages.page consumes, storage persists it as JSON) and has no
behavior that depends on adapters or runs.

What the translator reads (via `brief_slice` in stages.brief):
  - brief_prose    → free-form briefing prose from context agent
                     (tradition, genre, pacing, fallback guidance).
                     Injected verbatim into translator system prompt.
  - glossary       → source-token → target-rendering decisions
                     (e.g. `周妍 → Chu Nghiên`). Not identity-mapped.
  - address_pairs  → (speaker, listener) → pronoun pair phrase, e.g.
                     `("Tanaka", "Suzuki") → "tôi ↔ anh"`.
  - style_notes    → register/mood/translator-guidance lines (legacy;
                     superseded by brief_prose for new chapters).
  - key_notes      → per-bubble "Speaker: X → Y" hint
  - characters     → name + target_name + gender + role + voice

What the render path reads:
  - noise_keys / noise_pages → skip rendering

What material memory persists (not consumed by translator directly):
  - characters → full discovered character data (name, gender, role)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Character:
    """One character discovered by the vision pass.

    `gender` is one of "male" | "female" | "unknown". `role` is a short
    free-text descriptor ("young man", "elderly devil hunter", etc.).
    `voice` is a short character-voice descriptor for the translator
    ("cold", "rough", "polite", "teasing", "shy", "threatening"); empty
    when the agent cannot tell.
    `target_name` is the resolved rendering in the target language
    (e.g. "Chu Nghiên" for zh source → vi target). Falls back to `name`
    when the agent did not emit a target form.
    """
    name:        str
    target_name: str = ""
    gender:      str = "unknown"
    role:        str = ""
    voice:       str = ""

    @property
    def display_name(self) -> str:
        """Resolved target name, falling back to source name."""
        return self.target_name or self.name

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "target_name": self.target_name,
            "gender":      self.gender,
            "role":        self.role,
            "voice":       self.voice,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Character":
        return cls(
            name=d["name"],
            target_name=d.get("target_name", ""),
            gender=d.get("gender", "unknown"),
            role=d.get("role", ""),
            voice=d.get("voice", ""),
        )


@dataclass(slots=True)
class ChapterBrief:
    # Free-form briefing prose written by the context agent in the
    # target language. Injected verbatim into the translator's system
    # prompt. Contains: tradition inference, genre, pacing, SFX policy,
    # fallback register. Empty string when the context pass failed.
    brief_prose:   str                         = ""
    glossary:      dict[str, str]              = field(default_factory=dict)
    # (speaker_name, listener_name) → pronoun pair phrase, e.g.
    # "anh ↔ em" or "tôi ↔ ông". Speaker/listener names match
    # Character.name (source form). Special tokens "narrator" /
    # "unknown" are ignored by the translator.
    address_pairs: dict[tuple[str, str], str]  = field(default_factory=dict)
    style_notes:   list[str]                   = field(default_factory=list)
    key_notes:     dict[str, str]              = field(default_factory=dict)
    characters:    list[Character]             = field(default_factory=list)
    noise_keys:    set[str]                    = field(default_factory=set)
    # Page indices that are entirely non-diegetic (full-page credits,
    # ads, platform banners). These pages are dropped from the public
    # render archive entirely.
    noise_pages:   set[int]                    = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief_prose":   self.brief_prose,
            "glossary":      self.glossary,
            "address_pairs": [
                {"speaker": s, "listener": l, "pair": p}
                for (s, l), p in self.address_pairs.items()
            ],
            "style_notes":   self.style_notes,
            "key_notes":     self.key_notes,
            "characters":    [c.to_dict() for c in self.characters],
            "noise_keys":    sorted(self.noise_keys),
            "noise_pages":   sorted(self.noise_pages),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterBrief":
        return cls(
            brief_prose=d.get("brief_prose", ""),
            glossary=dict(d.get("glossary", {})),
            address_pairs={
                (item["speaker"], item["listener"]): item["pair"]
                for item in d.get("address_pairs", [])
            },
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
