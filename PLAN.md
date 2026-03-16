# Structured Character Profiles — Implementation Plan

## Problem

Current `add_note(type="character", content="free text")` system has 3 issues:

1. **Append-only** — no upsert. Same character gets multiple disconnected notes
2. **Dedup by exact match** — "Tanaka is quiet" and "Tanaka is a student" both survive
3. **Prompt bloat** — 2000 char budget fills fast with duplicate character info

Result: LLM gets fragmented, potentially contradictory character context → inconsistent xưng hô and voice.

## Solution

Add a dedicated `characters` table + `update_character` tool that **upserts by name**. LLM calls it whenever it learns something new about a character. The tool merges into the existing profile rather than appending a new row.

## Changes

### 1. Schema: new `characters` table in `context/db.rs`

```sql
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    alt_names TEXT DEFAULT '',          -- comma-separated aliases/source names
    age_group TEXT DEFAULT '',          -- child, teen, adult, elder
    gender TEXT DEFAULT '',             -- male, female, unknown
    speech_style TEXT DEFAULT '',       -- formal, casual, rough, polite, etc.
    description TEXT DEFAULT '',        -- free-text traits, appearance
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(project_id, name)
);

CREATE TABLE IF NOT EXISTS character_relationships (
    id INTEGER PRIMARY KEY,
    project_id TEXT NOT NULL,
    source_name TEXT NOT NULL,          -- speaker
    target_name TEXT NOT NULL,          -- addressee
    relationship TEXT NOT NULL,         -- siblings, classmates, boss-subordinate, etc.
    pronoun_pair TEXT DEFAULT '',       -- "anh/em", "tao/mày", etc. (target-lang specific)
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(project_id, source_name, target_name)
);
```

Key: `UNIQUE` constraint enables natural upsert via `INSERT OR REPLACE`.

### 2. New tool: `update_character` in `translation/tools/update_character.rs`

```rust
Args {
    name: String,              // canonical name
    alt_names: Option<String>, // aliases (source lang name, nicknames)
    age_group: Option<String>, // child | teen | adult | elder
    gender: Option<String>,    // male | female
    speech_style: Option<String>, // formal | casual | rough | polite | gentle | aggressive
    description: Option<String>,  // free-text traits
    relationships: Option<Vec<RelationshipArg>>,
}

RelationshipArg {
    target: String,       // other character name
    relation: String,     // siblings, classmates, etc.
    pronoun_pair: Option<String>, // anh/em, tao/mày, etc.
}
```

Tool description tells LLM: "Update or create a character profile. Call whenever you discover new info about a character — the profile will be merged, not replaced. Include relationships and pronoun pairs for Vietnamese."

### 3. DB methods in `context/db.rs`

- `upsert_character(project_id, name, fields...)` — INSERT OR REPLACE, merging non-empty fields only
- `upsert_relationship(project_id, source, target, relation, pronoun_pair)` — same pattern
- `get_all_characters(project_id)` → `Vec<CharacterProfile>`
- `get_all_relationships(project_id)` → `Vec<CharacterRelationship>`

Merge logic for `upsert_character`: read existing row first, only overwrite fields where the new value is non-empty. This way LLM can update just `age_group` without losing `speech_style`.

### 4. Prompt injection in `prompt.rs` + `chapter.rs`

Replace character/relationship notes in `fetch_previous_notes` with structured profiles:

```
Characters:
- Tanaka (teen, male, casual): confident student council president
  Alt: 田中
  → Yuki: siblings (anh/em)
  → Sensei: student→teacher (em/thầy)
  → Mob Boss: hostile (tao/mày)

- Yuki (teen, female, polite): quiet honor student
  Alt: ゆき
  → Tanaka: siblings (em/anh)
```

New method `PromptBuilder::character_profiles()` replaces the character/relationship portion of `notes_section()`.

### 5. Agent loop: register tool in `agent.rs`

Add `update_character` to tool dispatch, same pattern as existing tools.

### 6. Migration: keep `add_note` for event/setting

- `add_note` keeps `event` and `setting` types only
- Remove `character` and `relationship` from `add_note` enum — LLM uses `update_character` instead
- Update workflow instructions in `prompt.rs` to mention `update_character` in step 4

### 7. `fetch_previous_notes` in `chapter.rs`

- Remove `character`/`relationship` filtering from note injection
- Add new `fetch_character_profiles()` that queries `characters` + `character_relationships` tables
- Inject as structured block before notes section

## File changes summary

| File | Change |
|------|--------|
| `src/context/db.rs` | Add tables, upsert methods, query methods |
| `src/translation/tools/update_character.rs` | New file — tool def + handler |
| `src/translation/tools/mod.rs` | Register new tool |
| `src/translation/agent.rs` | Dispatch new tool |
| `src/translation/prompt.rs` | Add `character_profiles()` section |
| `src/translation/mod.rs` | Add `CharacterProfile` struct to request |
| `src/pipeline/chapter.rs` | Add `fetch_character_profiles()`, inject into request |
| `src/translation/tools/add_note.rs` | Remove character/relationship from enum |

## Context sub-agent integration

The context sub-agent (cheap model like haiku) already searches notes + translations via `get_context()`.
Character profiles are **not** searched via FTS5 — they're small enough to inject directly into the prompt.
However, the sub-agent's system prompt should be updated to know about character profiles:

- Sub-agent gains a new `list_characters` tool: returns all character profiles for the project
- When the translation agent calls `get_context("What is the relationship between X and Y?")`,
  the sub-agent can check both the characters table AND the notes FTS5 index
- This costs almost nothing (haiku is cheap) and gives the sub-agent complete context

### Who writes vs who reads

| Agent | `update_character` | Read profiles |
|-------|-------------------|---------------|
| Translation agent (expensive) | YES — sees images, understands context | YES — via prompt injection |
| Context sub-agent (cheap) | NO — read-only | YES — via `list_characters` tool |

The translation agent is the only writer because it's the one that actually sees the manga pages
and understands who the characters are, their age, relationships, etc.

## What this does NOT change

- Glossary system (unchanged — terms, not characters)
- Event/setting notes (still free-text via `add_note`)
- FTS5 search (characters table is small, direct query is fine)
- Context agent / get_context (still searches translations + notes, plus new list_characters)
- view_page / translate / search_glossary tools (unchanged)

## Edge cases

- **Same character, different names**: `alt_names` field + LLM uses canonical `name`
- **Relationship changes**: upsert overwrites — latest chapter wins
- **New project, no characters**: empty section, LLM discovers from scratch
- **Budget**: character profiles are compact (~100 chars each), much denser than free-text notes
