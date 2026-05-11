# Material architecture — at a glance

> **Status**: planning. See [`docs/rfc/material-architecture.md`](../rfc/material-architecture.md) for the full RFC.
> Current implementation still uses the `projects` schema this page
> supersedes; do not write new code against this layout yet.

This page is the **shipping target** for the Phase B re-architect.
Keep it short — the RFC carries the design rationale. This page is
the reference card you grep for when writing code on the
`feat/material-architect` branch.

## Entities

```
Material      one row per "manga" (any origin)
  Chapter     pages of the material; pages live remote or local
    Translation  one owner's run of pipeline against a chapter
                 for a target_lang; carries its own render archive
```

Identity rules:

- A **source-backed Material** is unique on `(source, upstream_ref)`.
  Two users importing the same HappyMH manga share the row.
- An **ext-captured** or **uploaded** Material is per-import.
  No cross-user dedup.
- A **Translation** is unique on `(chapter_id, owner_id, target_lang)`.
  Re-translate uses `POST /translate/{id}/redo`, never a second
  INSERT.

## Pipeline ownership

| Artifact | Keyed by | Why |
|---|---|---|
| Raw pages | Chapter (`pages_origin`, `prepared_locator`) | Shared across all translations of that chapter |
| `prepared.bnl` | Chapter | Same — prepare runs once |
| Bubbles, geometry, briefs, `translation_bubbles` | Translation | Each translation re-runs scan + LLM with its own glossary |
| `masks.npz` | Translation | Geometry detection is per-translation |
| `render.bnl` (public) | Translation | The user-facing archive |

## Routes

```
Discovery / consumption:
  /browse                                    sources hub
  /browse/$source                            source landing
  /manga/$source/$ref                        material detail (any origin)
  /manga/$source/$ref/chapter/$id            reader (Phase C)

User's library:
  /library                                   cross-source, chip filter

User's translations (advanced):
  /translate                                 list of user's translations
  /translate/$id                             single-translation editor
                                             (rich chapter list,
                                              selection bar, settings)
```

`/projects/*` does not exist after the refactor.

## API surface

```
POST  /api/material/import                   source-backed lookup or create
POST  /api/material/upload-init|finalize     user-uploaded material
GET   /api/material/{id}                     detail + chapters + per-chapter
                                             translation overlay
PUT   /api/material/{id}/bookmark            toggle

GET   /api/chapter/{id}/pages                page URLs for raw read

POST  /api/translate                         body {chapter_id, target_lang}
GET   /api/translate/{id}                    state + archive URL
POST  /api/translate/{id}/redo               re-run
PATCH /api/translate/{id}                    shared, settings
DELETE /api/translate/{id}
SSE   /api/translate/{id}/events             per-translation live events
```

## Capabilities (UI-derived)

For each `/manga/$source/$ref` view, the capabilities the user has
on this material:

```
read              always
bookmark          always (server-side toggle)
auto_translate    when source language != user target language
spawn_translation always (creates a translation owned by the user)
edit_material     when imported_by = user (uploads / ext captures only)
delete_material   when imported_by = user
share_translation when user owns ≥1 translation on this material
```

There are no per-material owner/share toggles for source-backed
materials — sharing is a property of translations.

## Worker

`tasks` table re-keys `(chapter_id, stage)` → `(translation_id, stage)`
except for the `prepare` stage which still keys by `chapter_id`
(prepare writes the chapter row, not the translation).

LISTEN channel payload changes from `<chapter_id>` to either
`<chapter_id>` (prepare) or `<translation_id>` (scan / translate /
render).

## Storage keys

```
c/{chapter_id}/prepared.bnl                  pipeline blob, per chapter
t/{translation_id}/masks.npz                 pipeline blob, per translation
render/{hmac(translation_id, salt)}.bnl      public, per translation
```

CDN cache busts on salt rotation.

## Hard rules

- Translation is the unit of work. The user spawns translations, not
  projects.
- Material is the unit of reading. Library and bookmark attach here.
- `pages_origin = 'remote'` means the manifest runtime fetches at
  read-time; no local copy. We never archive raw pages from manifest
  sources to our blob store.
- Sharing belongs to translations. A user can mark their translation
  shared without sharing the entire material.
- "Hội Mê Truyện" is a UI filter (`translation.shared = TRUE`), not a
  Material origin.

## Migration

See [`docs/rfc/material-architecture.md`](../rfc/material-architecture.md) §8.
Branch from `f8977dd`, single merge commit, drop & recreate DB on
flag day, salt rotation invalidates CDN cache.
