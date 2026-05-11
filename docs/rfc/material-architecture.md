# RFC — Material + Translation Architecture (Phase B Re-architect)

> **Status**: design draft. Not yet implemented. Created on
> commit `f8977dd` as the rollback point. Refactor branches start
> from there; reverting to that SHA undoes everything below.

## 1. Why this exists

Today the codebase has one entity for "a manga the user is working
on": `projects`. The schema treats it as a **long-running translation
endeavor over a whole series** — owner, source-target lang pair,
share flag, chapter list, glossary, render archives.

That model contradicts how the user base actually behaves:

- **Reader-leaning** — most opens are "I want to read this chapter,
  in my language, now". The user does not perceive a project.
- **Translate-on-demand** — when a chapter is unreadable raw, the
  user wants to spawn a single-chapter translation, not commit to
  managing the whole series.
- **Mixed origin** — material comes from manifest sources
  (HappyMH, MangaDex, OTruyen), from the browser extension capturing
  arbitrary sites, from manual upload of doujins / scans the user
  has on disk, occasionally from ad-hoc paste of a single image.

The `projects` abstraction forces every one of those into a
"translation workflow over a series" container that the user has
to create up-front, even when the user just wants to read one
chapter.

This RFC proposes splitting `projects` into two entities — **Material**
(the manga, source-agnostic) and **Translation** (a per-chapter,
per-target-lang artifact) — and consolidating every read/translate
surface around them.

## 2. Goals

1. **One mental model** — Material is the unit a user follows / saves /
   reads, regardless of where its pages live (source, ext capture,
   upload).
2. **Translation is an action, not a container** — spawn-able from a
   chapter row, not from a "create project" wizard.
3. **Many translations per chapter** — same raw chapter can host VN,
   EN, and a private user variant simultaneously, each with its own
   render archive and bubble text.
4. **Reading discovery** — when several translations exist for a
   chapter, the user sees the list and picks one (MangaDex scanlation
   pattern).
5. **Source-agnostic library** — `/library` works the same whether a
   material was scraped from HappyMH, captured via extension, or
   uploaded as a zip.
6. **No parallel old/new pipelines** — `projects` and `chapters` get
   deleted in the same PR that ships the new schema, after E2E parity.

Non-goals (deferred to later phases):

- Cross-user translation marketplace (rating, claiming, review).
- Per-chapter translation forking ("variant of @user2's VN with my
  edits").
- Soft-deleted history surfaces.
- Multi-target-lang on a single chapter for the **same owner**
  (each owner gets one translation per (chapter, target_lang)).

## 3. Mental model

```
Material        Identity of a manga, source-agnostic.
                  Origin: source-backed | ext-captured | uploaded
                  May be associated with one upstream
                    (source, upstream_ref) — unique when present.

  Chapter       A unit of pages inside a Material.
                  Pages live either in a remote source (raw URLs the
                  reader fetches via DA proxy) or in storage as
                  user-uploaded files.

    Translation  An owner's render of a chapter into a target lang.
                  Carries its own bubbles, geometry, briefs, glossary
                  binding, render archive, share flag.
                  Many translations per chapter possible.
```

Old: `Project -> Chapter -> (bubbles, geometry, translations, archive)`
New: `Material -> Chapter -> Translation -> (bubbles, geometry, archive)`

The whole "bubbles / geometry / translations_text / chapter_briefs"
graph that today is keyed by `chapter_id` migrates to being keyed
by `translation_id` — because in reality each translation **redoes**
its own OCR + geometry + LLM run; sharing across translations is a
future optimization, not a current capability.

## 4. Schema

### 4.1 New tables

```sql
-- ── Material ──────────────────────────────────────────────────────
-- Identity for a manga. Source-backed + non-source-backed share the
-- same table; `origin` discriminates and (source, upstream_ref) is
-- unique only when both are present (partial unique index).

CREATE TABLE materials (
    id              BIGSERIAL PRIMARY KEY,
    origin          TEXT NOT NULL CHECK (origin IN ('source','extension','upload')),

    -- Source-backed identity. NULL for ext + upload.
    source          TEXT,         -- 'happymh' | 'mangadex' | 'otruyen' | …
    upstream_ref    TEXT,         -- mangaUrl as the manifest runtime knows it

    -- Display snapshot — denormalized from the source at import time,
    -- refreshed when the source reports a change.
    title           TEXT NOT NULL,
    cover_url       TEXT,
    description     TEXT,
    author          TEXT,
    status          TEXT,
    languages       TEXT[] NOT NULL DEFAULT '{}',   -- source's available langs

    -- Provenance (who first imported / uploaded this).
    imported_by     BIGINT REFERENCES users(id) ON DELETE SET NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source-backed materials are deduped by (source, upstream_ref);
-- a second user importing the same manga reuses the row. ext + upload
-- materials are per-import (no dedup — each capture is its own thing).
CREATE UNIQUE INDEX uniq_materials_source_ref
    ON materials (source, upstream_ref)
    WHERE source IS NOT NULL;


-- ── Chapter ──────────────────────────────────────────────────────
-- A chapter inside a material. Pages either reference upstream URLs
-- (resolved at read-time via the manifest runtime) or live in our
-- pipeline storage (uploaded). Pure identity — no pipeline state.

CREATE TABLE chapters (
    id              BIGSERIAL PRIMARY KEY,
    material_id     BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,

    position        INTEGER NOT NULL,   -- sparse server-managed sort key
    number          TEXT NOT NULL,      -- display, free-form
    label           TEXT,               -- full label as the source presents it
    upstream_url    TEXT,               -- chapter URL on source; NULL for upload

    -- Source of the page pixels:
    -- 'remote' = fetch via manifest at read-time (no local storage)
    -- 'local'  = pages live in our blob store (upload + ext capture)
    pages_origin    TEXT NOT NULL CHECK (pages_origin IN ('remote','local')),

    -- For pages_origin = 'local'; the prepared archive key in the
    -- pipeline blob store (replaces today's prepared_key()).
    prepared_locator TEXT,
    prepared_backend TEXT,

    page_count      INTEGER NOT NULL DEFAULT 0,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (material_id, position)
);


-- ── Translation ─────────────────────────────────────────────────
-- The pipeline output for a (chapter, owner, target_lang) tuple.
-- Many translations per chapter possible. Owns its own bubbles,
-- briefs, render archive, share flag.

CREATE TABLE translations (
    id              BIGSERIAL PRIMARY KEY,
    chapter_id      BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    owner_id        BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_lang     TEXT NOT NULL,

    state           TEXT NOT NULL DEFAULT 'idle'
                       CHECK (state IN ('idle','pending','running','error','done')),
    stage           TEXT,                 -- current pipeline stage if running

    archive_backend TEXT,
    archive_locator TEXT,
    rendered_at     TIMESTAMPTZ,
    progress_stage  TEXT,
    progress_index  INTEGER,
    progress_total  INTEGER,

    shared          BOOLEAN NOT NULL DEFAULT FALSE,
    settings        JSONB,                 -- inherit-from-defaults override

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (chapter_id, owner_id, target_lang)
);
CREATE INDEX idx_translations_chapter      ON translations (chapter_id);
CREATE INDEX idx_translations_owner        ON translations (owner_id);
CREATE INDEX idx_translations_shared_chapter
    ON translations (chapter_id) WHERE shared = TRUE;
```

### 4.2 Migrated tables (re-keyed from chapter_id → translation_id)

The pipeline output graph today hangs off `chapter_id`. Each owner +
target_lang gets its own copy after the refactor, so the FK shifts.

```sql
-- bubbles, bubble_geometry, page_geometry, translations (text),
-- chapter_briefs, tasks, chapter_consumes — all change their FK from
-- chapter_id to translation_id. Schema is otherwise unchanged.
--
-- Naming collision: today's `translations` table stores per-bubble
-- translated text. Rename to `translation_bubbles` to free the noun.

ALTER TABLE translations RENAME TO translation_bubbles;
-- (then create the new translations table per §4.1)

ALTER TABLE bubbles            RENAME COLUMN chapter_id TO translation_id;
ALTER TABLE bubble_geometry    RENAME COLUMN chapter_id TO translation_id;
ALTER TABLE page_geometry      RENAME COLUMN chapter_id TO translation_id;
ALTER TABLE translation_bubbles RENAME COLUMN chapter_id TO translation_id;
ALTER TABLE chapter_briefs     RENAME COLUMN chapter_id TO translation_id;
-- (chapter_briefs name stays — it's still a brief, scoped to one
--  translation's run)
```

In practice this is a **schema rebuild** (DDL-only DB per
`schema.sql:6`), not an in-place migration. See §8.

### 4.3 Bookmark + library

Today: `project_pins` keyed by `(user_id, project_id)`.

After: `material_pins` keyed by `(user_id, material_id)`. Bookmark
attaches to the material (the manga as a whole), not to a particular
translation — same as Phase A library on the frontend.

```sql
CREATE TABLE material_pins (
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    material_id BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    pinned_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, material_id)
);
```

### 4.4 Glossary

Glossary today is per-project. After the refactor, glossary is
**per-user-per-source-lang** (a user dicts terms once, every
translation they own uses it). This collapses a per-project N×
duplication into one row set per (user, source_lang).

```sql
CREATE TABLE glossary (
    id           BIGSERIAL PRIMARY KEY,
    owner_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    notes        TEXT,
    UNIQUE (owner_id, source_lang, target_lang, source_term)
);
```

Per-translation override is possible but not in scope for the
initial cut.

### 4.5 Quota

`chapter_consumes` shifts to consume a **translation_id** rather than
a chapter_id. Counter logic unchanged.

## 5. Storage layout

`archive_token(project_id, chapter_id)` becomes
`archive_token(translation_id)`. Archive paths:

```
Old:
  p/{project_id}/c/{chapter_id}/prepared.bnl   (pipeline blob)
  p/{project_id}/c/{chapter_id}/masks.npz      (pipeline blob)
  render/{hmac(project_id, chapter_id)}.bnl    (public)

New:
  c/{chapter_id}/prepared.bnl                  (pipeline blob, per chapter)
  t/{translation_id}/masks.npz                 (pipeline blob, per translation)
  render/{hmac(translation_id)}.bnl            (public, per translation)
```

`prepared.bnl` stays per-chapter — the prepared pages are the raw
content, shared across every translation that runs on the chapter.
`masks.npz` becomes per-translation because masks evolve with the
geometry detection pass which each translation re-runs.

Render archive HMAC takes only `translation_id` + salt. The CDN
cache key changes for every chapter — this **invalidates the
existing CDN cache** entirely. Plan for that: section §8.4.

## 6. Pipeline / worker

`tasks` re-keys from `(chapter_id, stage)` to `(translation_id, stage)`.
The trigger that wakes workers via `pg_notify` carries
`translation_id` in payload. Worker dispatch logic shrinks because
the worker no longer needs to look up the owning project — every
piece of state it needs (target_lang, glossary scope, owner_id) is
on the translation row.

### 6.1 Sharing prepared pages

`stages/prepare.py` runs once per chapter and writes
`prepared.bnl`. Subsequent translations on the same chapter skip
prepare. Trigger:

```
on POST /translate/chapter:
  ensure translation row (chapter, owner, target_lang)
  if chapter.prepared_locator IS NULL:
      enqueue task (translation, 'prepare')   -- writes chapter.prepared_locator
      after prepare done → enqueue 'scan'
  else:
      enqueue task (translation, 'scan')
```

`prepare` is the only stage that mutates the **chapter** row.
Every later stage writes to **translation**. This means N
translations on the same chapter share the cost of prepare exactly
once.

### 6.2 Concurrency

Two users translating the same chapter to the same lang? Disallow at
DB level (`UNIQUE (chapter_id, owner_id, target_lang)` already does
this for same owner). Two users different lang? Both run, both pay
their own quota, both write their own pipeline output to
`t/{translation_id}/...`. No coordination needed.

## 7. API surface

### 7.1 Material discovery & import

```
GET  /api/material/{source}/{upstream_ref_b64}    lookup or 404
POST /api/material/import                         create from manifest+ref
POST /api/material/upload-init                    user-uploaded zip → init
POST /api/material/upload-finalize                ... → finalize, create chapters
```

Source-backed import (`/import`) is idempotent — second call with
the same (source, upstream_ref) returns the existing row. This is
what makes the manga-card click flow "open detail" rather than
"create".

### 7.2 Chapter read

```
GET  /api/material/{id}                           material detail
GET  /api/material/{id}/chapters                  list (+ overlay per-chapter
                                                  translation summary)
GET  /api/chapter/{id}/pages                      page URLs for raw read
                                                  (manifest dispatch for source-
                                                   backed; storage for local)
```

The chapter list endpoint embeds a small array per chapter:

```json
{
  "id": 4711,
  "number": "1100",
  "translations": [
    {"id": 9001, "owner": "@me",       "lang": "vi", "state": "done", "shared": false},
    {"id": 9002, "owner": "@nghyane",  "lang": "vi", "state": "done", "shared": true},
    {"id": 9003, "owner": "@user2",    "lang": "en", "state": "running", "shared": true}
  ]
}
```

Frontend can render the chapter row directly — "raw" link + one button
per translation. No second round-trip.

### 7.3 Translate

```
POST /api/translate                               body: {chapter_id, target_lang}
                                                  returns translation_id
GET  /api/translate/{id}                          state, progress, archive URL
POST /api/translate/{id}/redo                     re-run pipeline
PATCH /api/translate/{id}                         shared, settings
DELETE /api/translate/{id}                        cascades archive cleanup
SSE  /api/translate/{id}/events                   per-translation event stream
```

Project-level routes go away. There's no "create translation
collection" — the user spawns translations per chapter; the user's
collection is implicit (`SELECT * FROM translations WHERE owner_id = ?`).

### 7.4 Library

```
GET  /api/library                                 materials user has touched
                                                  (bookmark + history union;
                                                  history lives client-side
                                                  but bookmark moves server-side)
PUT  /api/material/{id}/bookmark                  toggle
```

The local-first library store from Phase A stays — but bookmark
becomes a server write (mirror'd back to local cache) because it's
the only state that needs cross-device sync in this phase. Reading
history remains local-only.

## 8. Migration plan

### 8.1 Branching strategy

```
main (today)
  ├─ checkpoint f8977dd                ← rollback point, this commit
  └─ feat/material-architect           ← new branch
       ├─ feat/material-schema         ← schema.sql + postgres adapter rewrite
       ├─ feat/material-api            ← routes, pydantic models, deps
       ├─ feat/material-pipeline       ← worker re-keying, prepare/scan/render
       ├─ feat/material-storage        ← archive key shape, CDN purge
       ├─ feat/material-web            ← /manga/$source/$ref, ChapterRow,
       │                                   library re-targeting
       └─ feat/material-ext            ← ext rewrite: /api/translate flow
```

Each sub-branch merges into `feat/material-architect`, never into
`main`. The full branch lands as **one merge commit** after E2E
parity passes. Rollback = `git revert <merge-sha>` OR
`git reset --hard f8977dd`.

### 8.2 Data migration

Per `schema.sql:6` the convention is **DDL-only, no in-place
migration**: `dropdb && createdb`. That's tenable for Phase 1 because
the existing prod DB is a beta with a small user base.

If we need to preserve current projects (decision needed):

```
For each project p:
  insert material (
    origin = 'source' if p.source_url matches a known manifest
            else      'upload',
    source/upstream_ref = derived from p.source_url when source,
    title/cover/desc = p's fields,
    imported_by = p.owner_id
  )
  For each chapter ch in p:
    insert chapter (material_id = …, position, number, …,
                    pages_origin='local',
                    prepared_locator = old prepared_key())
    insert translation (chapter_id = …, owner_id = p.owner_id,
                        target_lang = p.target_lang, state = ch.state
                        archive_backend/locator = ch.archive_*)
    move bubbles/geometry/translation_bubbles/briefs/tasks rows
      from chapter_id to translation_id
```

Decision needed before implementation:

- **Drop & recreate** (clean, fast, loses everything) — viable if beta
  user count is low and pre-announce a wipe.
- **In-place migrate** (preserves data, ~200 LOC of one-shot SQL +
  blob renames in R2/HF, painful) — only if user count justifies.

Recommendation: **drop & recreate** for the beta; revisit when there's
a real user base. AGENTS.md hard rule "delete dead legacy code" backs
this — leaving migration scripts around invites confusion.

### 8.3 Backward URL compat

Old routes:

```
/projects/$projectId            → drop. No redirect.
/projects/$projectId/chapters/* → drop. No redirect.
/browse/$source/manga/$mangaId  → 301 → /manga/$source/$mangaId
```

The browse → manga redirect is cheap (TanStack Router static map);
adding it costs nothing and preserves bookmarks during the beta.
Project URL redirect is impossible (the mapping project_id →
material_id only exists post-migration); we don't try.

### 8.4 CDN cache invalidation

The `archive_token` HMAC changes for every chapter (input went from
`(project_id, chapter_id)` to `(translation_id)`). Every render URL
the browser has cached, every `?v=` token in `chapters.rendered_at`,
becomes a 404.

Mitigations:

- **Rotate the salt** (already supported per `archive_token` doc).
  This forces all clients to discard cached URLs on next manga page
  open. Cost: zero — `?v=` tokens already exist as the cache-bust
  affordance.
- **Pre-warm** popular materials' renders into the new key space
  before flipping the SPA. Not worth the complexity for a beta.

### 8.5 Extension migration

Extension today: project picker → upload chapter zip to selected
project. After: extension calls `/api/material/import` with
`(source = host of current tab, upstream_ref = page URL)`, gets
back a material_id, then `/api/translate { chapter_id, target_lang }`
per chapter the user captured.

For non-manifest sites the extension uploads pages to
`/api/material/upload-finalize` with `pages_origin = 'local'` and a
synthesized `material` row (origin='extension'). Chapters get
created on the way in.

Ship the ext rewrite **in the same PR** as the web rewrite. AGENTS.md
forbids parallel old/new paths.

### 8.6 Order of operations on the merge day

1. Code-freeze the rollback branch.
2. `wrangler pages deploy` an SPA build that points at the OLD API
   (= current production, no change).
3. Apply schema migration (drop + recreate, or one-shot migration
   script) on the production DB.
4. Restart `com.typoon.api` + `com.typoon.worker` LaunchAgents on
   the new code.
5. Rotate `BLOB_SALT` (CDN cache invalidation §8.4).
6. `wrangler pages deploy` the new SPA build.
7. Smoke-test (deploy-beta.md §7 probe list, plus new probes:
   `GET /api/material/.../chapters`, `POST /api/translate`).
8. Tag the merge SHA. Document in this RFC: "shipped at <SHA>".

Rollback procedure if step 7 fails:

```
git reset --hard f8977dd
launchctl kickstart -k gui/$UID/com.typoon.api
launchctl kickstart -k gui/$UID/com.typoon.worker
wrangler pages deploy web/dist --project-name=mangalocal-web --branch=main
# (using web/dist built from f8977dd)
psql typoon -f checkpoint-schema.sql   # the dump taken in step 0
```

Take the schema + data dump in step 0 (above step 1) so the rollback
actually works:

```
pg_dump --format=custom typoon > checkpoint-$(date +%s).dump
```

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pipeline regression on prepared.bnl sharing | M | H | Single-translation case (the only one today) is identical to old behavior; multi-translation is new and gets E2E tests before merge |
| User data loss on drop & recreate | M | M | Take pg_dump + R2 inventory before flip; revert restores both |
| CDN cache stale → 404 storm on first hour after flip | H | L | Salt rotation kills cached URLs cleanly; old archives still exist at new keys after re-render — users hit a fresh render once |
| ext rewrite shipped late | M | H | Ext is in `ext/` repo subtree; not blocking — if behind, disable ext button in popup until it lands |
| Glossary collapse from per-project to per-user-per-lang surprises owners | L | M | Phase 1 migrate: every project's glossary becomes the owner's glossary for that lang pair (UNION). Owners with multiple projects on same pair get a merged glossary; document in CHANGELOG |
| `UNIQUE (chapter, owner, target_lang)` blocks re-translate flow | L | M | UI uses `POST /translate/.../redo` (idempotent on the same row); never a second INSERT |
| Worker queue confusion during transition | M | M | Flag day deploy (§8.6) keeps old and new schemas from ever coexisting |
| Reading-history merge with bookmark on backend | L | L | History stays client-only this phase; only bookmark moves server-side |

## 10. What stays the same

- Discord auth, JWT minting, API tokens, RFC-008 / RFC-009.
- Render archive format (Bunle / `.bnl`).
- Worker LISTEN/NOTIFY pattern; only payload keys change.
- Frontend manifest runtime (`features/browse/manifest/*`) — Material
  source-backed import calls the runtime exactly as today.
- Browse hub, source picker, shelves, search, infinite scroll —
  unchanged.
- Library (`features/library/store.ts`) — local-first, store shape
  expands to track material_id alongside (source, ref) but writes
  through to `/api/material/.../bookmark` for the bookmark flag only.
- DA setup, deploy topology, CORS, TRUSTED_HOSTS — unchanged.

## 11. What gets deleted

After merge, the following disappear (AGENTS.md doctrine):

```
typoon/api/routes/projects.py        → split into material.py +
                                       chapter.py + translate.py
typoon/api/routes/project_events.py  → translate_events.py
typoon/adapters/projects.py          → split into material.py +
                                       translate.py
web/src/features/project-detail/     → merged into features/manga/
web/src/routes/projects.*.tsx        → deleted, replaced by
                                       /manga/$source/$ref
typoon/storage/schema.sql            → rewritten in place
```

Wiki pages to update:

```
docs/wiki/architecture.md           — package layout reflects
                                      material/ + translate/
docs/wiki/browse-mode.md            — §1, §5, §11 (Material is the
                                      unit; Hội Mê Truyện is a filter
                                      not a source)
docs/wiki/render-archive-storage.md — key path layout, salt rotation
docs/wiki/deploy-beta.md            — probe list, flag day playbook
docs/wiki/hard-rules.md             — strike "project pipeline";
                                      add "translation is the unit"
```

## 12. Out of scope (Phase C+)

- Chapter reader unification — internal vs external readers still
  fetch differently (.bnl vs raw images via manifest). Same hero,
  different page loader. Phase C tackles it.
- Translation rating / claim flow ("I want to translate this").
- Per-chapter glossary override.
- Watch Party (DA-only shared reading session).
- Discord rich presence ("Reading X · chapter Y").
- Background new-chapter refresh worker.
- Cross-device history sync.

## 13. Open questions

1. **Drop-or-migrate data?** §8.2 recommends drop; need decision.
2. **Translation share visibility** — is it per-translation
   (each owner gates their own) or per-material (owner aggregate)?
   Default: per-translation. The owner of an unshared material can
   still have a shared translation on its chapter.
3. **Material identity for upload** — if two users upload the same
   doujin separately, we currently get two materials. Should we
   dedup by content hash? Default: no; user expectation is that
   their upload is private to them until explicitly shared.
4. **Auto-spawn translation on chapter capture** — should ext
   automatically `POST /translate` after capture, using the user's
   default target_lang? Default: yes for ext (one-tap UX); no for
   `/api/material/upload` (user might be uploading a re-read).
5. **`source = 'community'` synthetic value** — does it still exist?
   Default: no. "Hội Mê Truyện" is a filter
   (`WHERE translation.shared = TRUE`) over the unified material
   table, not a synthetic source row.

Answer the five before §8.6 flag day.
