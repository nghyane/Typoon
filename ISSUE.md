# Issue 1: Character context is append-only — causes fragmentation and inconsistent xưng hô

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

---

# Issue 2: No way to seed context from existing translations — cold start for every series

## Problem

When starting a new series that already has professional translations available (official Vietnamese releases, fan translations, etc.), the system starts from zero. The LLM has no reference for:

- **Established character names** — how were they localized?
- **Xưng hô conventions** — what pronoun pairs were used for each character relationship?
- **Tone and register** — is the series casual, formal, comedic, dark?
- **Terminology** — recurring terms, attack names, place names already have accepted translations

The `ContextStore` already has the right structure for this — `translations` table stores `source_text → translated_text` per bubble, FTS5 indexes it, and the context sub-agent (haiku) already searches it via `get_context()`. The `Glossary` also supports `import_toml()` for term import.

But there's no way to **feed existing translations in** without running them through the full pipeline.

## What's missing

### 1. No import path for translation pairs

`save_chapter()` exists but is only called internally after the LLM translates. There's no CLI command or external interface to bulk-import `source → translated` pairs from an existing translation.

### 2. No way to seed character/relationship context

Even if translation pairs were imported, the character notes wouldn't exist. The LLM builds these incrementally via `add_note()` — but for an imported series, nobody calls `add_note()`.

### 3. Glossary import exists but is disconnected

`import_toml()` handles terms, but character names + relationships are a different category. A professional translation of volume 1-5 contains a wealth of character context that should inform volume 6's translation.

## Why this matters

- **Consistency with existing translations** — readers who've read official volumes 1-5 expect volume 6 to match
- **Xưng hô accuracy from chapter 1** — instead of the LLM guessing and building context over time, it starts with established conventions
- **Reduced LLM research overhead** — fewer `view_page()` and `get_context()` calls needed when the system already knows the characters

## The architecture already supports this

| Component | Import capability | Status |
|-----------|------------------|--------|
| `translations` table | `save_chapter()` writes source→translated pairs | Exists, just needs external caller |
| `chapter_notes` table | `add_note()` writes free-text notes | Exists, needs bulk import |
| `glossary` table | `import_toml()` upserts terms | Exists and works |
| FTS5 indexes | Auto-sync via triggers | Already works |
| Context sub-agent | Searches all of the above | Already works |

The gap is purely an **import interface** — the storage and retrieval layers are ready.
