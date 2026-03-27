# Issue: Context layer needs a unified knowledge architecture

## Current state

Knowledge about a series is scattered across 3 disconnected stores with inconsistent semantics:

```
┌─────────────────────────────────────────────────────┐
│ context.db (ContextStore)                           │
│  ├── translations    append-per-chapter, immutable  │
│  └── chapter_notes   append-only, 4 types mixed     │
│       ├── character  ← should be upsert             │
│       ├── relationship ← should be upsert           │
│       ├── event      ← append is fine               │
│       └── setting    ← append is fine               │
├─────────────────────────────────────────────────────┤
│ glossary.db (Glossary)                              │
│  └── glossary        upsert, term→translation       │
└─────────────────────────────────────────────────────┘
```

### The problems

**1. `chapter_notes` is a catch-all for data with different lifecycles**

- Characters and relationships **accumulate and evolve** — they need upsert (merge new info into existing)
- Events and settings are **ephemeral/historical** — append is correct for them
- Mixing both in one table with identical schema means characters get the append treatment

**2. Two separate databases, two separate systems**

- `ContextStore` and `Glossary` are opened independently in `runner.rs`
- They share no data — glossary doesn't know about characters, context doesn't know about terms
- The translation agent gets both injected via different prompt sections (`glossary_section()` vs `notes_section()`) with no coordination
- The context sub-agent can only search `ContextStore`, not glossary

**3. No import path**

- `Glossary` has `import_toml()` — good
- `ContextStore` has `save_chapter()` but it's internal-only, no external import
- No way to seed characters, relationships, or translation pairs from existing translations
- Every series starts cold

**4. Prompt injection is fragmented**

`PromptBuilder` assembles knowledge from multiple disconnected sources:

```rust
// prompt.rs — each pulls from a different place
pub fn user_prompt(&self) -> String {
    ...
    prompt.push_str(&self.glossary_section());    // ← from Glossary DB
    prompt.push_str(&self.notes_section());       // ← from ContextStore notes
    prompt.push_str(&self.previous_translations()); // ← from caller-provided context
    ...
}
```

And `chapter.rs` does its own filtering/budgeting in `fetch_previous_notes()`:
- Only injects `character` + `relationship` notes (hard-coded filter)
- Dedup by exact content match (misses same-character-different-wording)
- 2000 char budget shared across all note types

**5. Tools reflect the fragmentation**

Translation agent has separate tools that write to different stores:
- `add_note()` → `chapter_notes` table (append)
- `update_glossary()` → `glossary` table (upsert)
- No tool for structured character data

## What a clean architecture looks like

The knowledge a translation system needs about a series falls into clear categories:

| Data type | Lifecycle | Current store | Correct behavior |
|-----------|-----------|---------------|-----------------|
| **Characters** | Accumulate + evolve | `chapter_notes` (append) | Upsert by name |
| **Relationships** | Accumulate + evolve | `chapter_notes` (append) | Upsert by pair |
| **Terms/Glossary** | Stable, canonical | `glossary` (upsert) | Upsert — already correct |
| **Translations** | Historical record | `translations` (append) | Append — already correct |
| **Events** | Ephemeral/chapter-scoped | `chapter_notes` (append) | Append — already correct |
| **Settings** | Semi-persistent | `chapter_notes` (append) | Append — already correct |

The refactor should:

1. **Give characters and relationships their own tables** with upsert semantics and structured fields (name, age_group, gender, speech_style, pronoun_pairs)
2. **Unify into one DB** — glossary, characters, relationships, translations, notes all in one `context.db` so the sub-agent can search everything
3. **One import interface** — `comicscan import <project_id> <path>` that can seed all data types from existing translations
4. **Clean prompt assembly** — `PromptBuilder` pulls from one knowledge store with clear priority: characters → relationships → glossary → recent notes, each with its own budget

## Impact on existing code

### What stays the same
- `translations` table — append per chapter, works fine
- `event`/`setting` notes — append-only, works fine
- Translation agent loop — tool dispatch pattern unchanged
- Context sub-agent pattern — cheap model searching DB, unchanged
- Detection/OCR/Render pipeline — completely unaffected

### What changes
- `chapter_notes` loses `character`/`relationship` types → new dedicated tables
- `Glossary` merges into `ContextStore` (one DB, not two)
- New `update_character` tool replaces `add_note(type="character")`
- `add_note` keeps only `event` and `setting`
- `PromptBuilder` gets `character_profiles()` section with structured format
- `fetch_previous_notes()` simplified — only fetches events/settings
- New CLI command for importing existing translations + character data
- Config: `glossary.db_path` deprecated → glossary lives in `context.db`

## Context

Analyzed [thang97-21/MTLS](https://github.com/thang97-21/MTLS) — their system pre-resolves all character/relationship/term data before translation, injecting structured registries into the prompt. Different domain (light novels) but the core insight is the same: **character identity and relationships are not notes, they're structured data that evolves over time.**
