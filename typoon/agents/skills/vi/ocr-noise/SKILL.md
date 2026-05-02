---
name: ocr-noise
description: Use this skill when OCR noise is ambiguous or attached to real text while translating into Vietnamese: detached digits or symbols inside dialogue, cut-off fragments, repeated artifacts, or tokens that look like stats/codes but may belong to another panel. Load it before deciding whether suspicious tokens should be preserved, removed, or the whole bubble skipped. Do not load it for clean dialogue or obvious standalone noise already covered by basic skip rules.
---

# OCR noise filtering into Vietnamese

Use this skill for ambiguous OCR cases where suspicious text may be attached to real dialogue or may belong to another panel.

## Policy

- Preserve a number, symbol, code, timer, stat, or fragment only when it is clearly part of the intended dialogue, narration, sign, UI panel, or system message.
- Remove suspicious tokens inside dialogue when they are detached from the sentence, visually out of place, repeated, cut off, or inconsistent with surrounding text.
- Mark the whole bubble as `skip` when it is only a page marker, isolated digit/letter, random symbols, watermark, credit, URL, or scanner artifact.
- Do not force preservation just because a token contains digits or punctuation.
- If preserving a suspicious token makes the Vietnamese sentence unnatural and no context proves it belongs, remove it.

## Gotchas

- A real stat/code can become noise if OCR attached it to the wrong bubble.
- A short token near a panel edge is often a crop artifact or page marker, not dialogue.
- Prefer a clean natural Vietnamese sentence over preserving detached OCR junk.
