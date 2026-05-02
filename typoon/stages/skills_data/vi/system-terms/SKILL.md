---
name: system-terms
description: Use this skill when translating game/system/fantasy terminology into Vietnamese: skill names, named attacks, buffs, debuffs, ability activations, system messages, UI labels, stats, ranks, quests, item names, status windows, or power names. Load it to decide whether terms should be rendered in natural Vietnamese, Han-Viet style, transliterated, preserved, or added to glossary. Do not load it for ordinary dialogue, character names, places, or non-system text with no special terms.
---

# System terms into Vietnamese

Use this skill for game-like or fantasy system terminology: named powers, attacks, buffs, debuffs, system messages, UI labels, ranks, quests, items, stats, and status windows.

## Default policy

- Translate readable natural-language terms into Vietnamese by default.
- Use compact Vietnamese suitable for speech bubbles, system windows, and ability names.
- Prefer a polished fantasy/action register when the term is a named power or rank.
- Preserve the source form only when it is a fixed identifier, acronym, branded/proper name, glossary-locked term, or visually part of unchanged UI art.
- Keep one Vietnamese rendering for recurring terms and add important recurring terms to the glossary.
- If the term is visually or syntactically suspicious, apply `ocr-noise` before deciding the final rendering.

## Gotchas

- Do not keep English merely because the source uses Title Case.
- Do not translate character names, faction names, or place names as if they were skills.
- Do not preserve OCR fragments inside a term.
- If a term appears in a system window, keep it concise and consistent.
- If only tone/intensity is at issue and terminology is already clear, use `action-tone` rather than this skill.
