# Character context is append-only — causes fragmentation and inconsistent xưng hô

## Problem

The current `add_note(type="character")` system stores character information as append-only free-text entries in `chapter_notes`. Over multiple chapters, this causes:

### 1. No upsert — same character accumulates disconnected notes

Each chapter appends a new row. After 10 chapters, a single character like Tanaka might have:

```
Ch.1: [character] Tanaka is a quiet student
Ch.3: [character] Tanaka seems to be class representative
Ch.5: [character] Tanaka is now confident after the tournament
Ch.8: [relationship] Tanaka and Yuki are siblings
Ch.12: [character] Tanaka is the student council president
```

No single source of truth — just fragments.

### 2. Dedup fails for same-character notes

`fetch_previous_notes` in `chapter.rs` deduplicates by **exact content match**. Two notes about the same character with different wording both survive. The LLM gets redundant, sometimes contradictory context.

### 3. Prompt bloat → budget exhaustion

Character + relationship notes share a 2000-char budget (`NOTES_BUDGET_CHARS`). With fragmented notes, this budget fills up fast — pushing out newer, more relevant information in favor of older entries.

### 4. Inconsistent xưng hô (Vietnamese pronouns)

Vietnamese translation depends heavily on character relationships for pronoun selection (anh/em, tao/mày, cậu/tớ, etc.). Without a structured relationship map, the LLM may choose different pronouns for the same character pair across chapters.

## Impact

- Translation quality degrades over longer series (more chapters = more noise)
- Xưng hô inconsistency is the most visible artifact for Vietnamese readers
- The 2000-char budget becomes a bottleneck instead of a reasonable limit

## Context

Analyzed [thang97-21/MTLS](https://github.com/thang97-21/MTLS) which uses structured character registries with voice fingerprints and relationship maps — their profiles are compact and always up-to-date. Different domain (light novels vs manga) but the fragmentation problem is the same.
