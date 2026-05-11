# RFC — Material + Translation Architecture (v5 — community-driven identity)

> **Status**: design final. Implementation begins from commit `f8977dd`
> (the rollback point). Reverting to that SHA undoes everything below.
>
> Changelog from v4:
> - **Material cross-user dedup** for source-backed manga (replaces
>   per-user material rows). Privacy isolation came from
>   per-translation visibility, not material identity; per-user
>   material was overcautious.
> - **Community link voting** as the cross-source identity strategy.
>   Replaces fuzzy title match, cover hash, and external MangaDex API.
>   User actions (link / unlink / confirm / reject) accumulate votes
>   on `material_link_votes`; suggestions derive from accumulated
>   score.
> - **Manifest extended** with `title_native`, `title_alt`,
>   `cross_refs` to seed identity hints (1 line of selector work per
>   source).
> - **Cold-start strategy**: admin seeds top-100 manga links on
>   launch; manifest hints carry the rest; community voting accrues
>   organic over time.
>
> Changelog from v3 (carried forward):
> - DA-only (no plain web mode)
> - Guild-scoped sharing (`visibility ∈ {private, guild, all_guilds}`)
> - Auto-share opt-out (default share) replacing manual opt-in
> - CAS via `chapter.prepared_hash`
> - 3-layer pipeline ownership (chapter / draft / translation)
> - Community glossary scaffold
> - DMCA takedown procedure with per-guild scope
> - Removed `projects`, `project_pins`, `material_pins`, `shared` bool

## 1. Why this exists

Today the codebase has one entity for "a manga the user is working
on": `projects`. The schema treats it as a long-running translation
endeavor over a whole series — owner, source-target lang pair, share
flag, chapter list, glossary, render archives.

That model contradicts user behavior:

- **Reader-leaning** — most opens are "I want to read this chapter,
  in my language, now". The user does not perceive a project.
- **Translate-on-demand** — when a chapter is unreadable raw, the
  user spawns a single-chapter translation, not a whole-series
  workflow.
- **Mixed origin** — material comes from manifest sources, browser
  extension captures, manual zip uploads, occasionally ad-hoc paste.
- **DA-only deployment** — app runs exclusively inside Discord
  Activity. Identity is Discord; audience scope is Discord guilds;
  no SEO surface; no plain-web access path. The `projects` model
  doesn't leverage any of this.

This RFC splits `projects` into **Material** (the manga, source-
agnostic) and **Translation** (a per-chapter, per-target-lang
artifact). Sharing is guild-scoped (not public). Caching is
content-addressable across users in shared guilds. No `projects`
left after merge.

## 2. Goals

1. **One mental model** — Material is the unit a user follows / saves /
   reads, regardless of origin (source, ext capture, upload).
2. **Translation is an action** — spawned from a chapter row, not
   from a "create project" wizard.
3. **3-layer pipeline ownership** — prepare and scan shared across
   all translations of a chapter; draft shared across translations
   of (chapter, lang_pair, glossary_fp); render shared per draft
   when no edits. Same translation request from 100 users in the
   same guild costs ~1 LLM call instead of 100.
4. **Guild-scoped sharing** — Hội Mê Truyện is a per-guild feed.
   No global public feed. DMCA exposure is per-guild.
5. **Library entries** group multiple Materials representing the
   same manga (per-user, optionally fuzzy-matched).
6. **DA-only** — no plain-web flow, no anonymous access, no SEO
   landing. Auth is Discord OAuth via SDK. Discord identity and
   guild memberships drive every access decision.
7. **No parallel old/new pipelines** — `projects` and its dependents
   get deleted in the same PR that ships the new schema, after E2E
   parity.

Non-goals (deferred):

- Cross-user translation marketplace (rating, claiming, review).
- Per-chapter translation forking ("variant of @user2's VN with my
  edits"). Sparse edits over a shared draft cover 80% of this.
- Soft-deleted history surfaces.
- External ID matching (MangaDex universal ID for cross-source
  identity).
- Watch Party shared-scroll reading.
- Discord Rich Presence (phase 2).
- Guild Bot embed notifications (phase 2).
- Public web access. Permanent.

## 3. Mental model

```
Material           Identity of a manga, source-agnostic.
  origin ∈ {source, extension, upload}
  Source-backed materials are CROSS-USER: two users importing
    HappyMH/Naruto share the same material row (unique on
    (source, upstream_ref)). `imported_by` tracks the first user
    as an audit trail; ownership is NOT enforced here.
  Extension / upload materials remain per-user (no upstream_ref to
    dedup against; each capture is its own identity).

Chapter            A unit of pages inside a Material.
  pages_origin ∈ {remote, local}
  remote: pages fetched at read-time via manifest runtime.
  local:  prepared.bnl lives in our blob store.
  CAS via prepared_hash deduplicates pixel content across chapters
    that happen to be byte-identical (e.g. same upload zip).

TranslationDraft   LLM output bound to (chapter, source_lang, target_lang,
                   glossary_fingerprint). Shared with visibility scope.
  visibility ∈ {private, guild, all_guilds}
  Cache hit when: same chapter, same lang pair, same glossary signature.

Translation        Per-user wrapper around a draft.
  Owns the render archive (or shares draft's default archive).
  Holds sparse edits (translation_edits) if user overrode bubbles.
  Carries user-facing flags (in_feed for Hội Mê Truyện inclusion).

LibraryEntry       Per-user grouping of one or more Materials that
                   represent the same manga from the user's POV.
  Bookmark, last-read tracking, "Continue reading" hang off here.
  Cross-source linking is community-voted (material_link_votes);
  suggestions surface to the user; user actions accumulate votes
  passively (link = +1, reject = -1, unlink = 0).
```

## 4. Schema

The DDL is the source of truth (`typoon/storage/schema.sql`). This
section sketches; refer to the migration commit for the actual file.

### 4.1 Identity

```sql
-- users + identities unchanged from current schema.
-- New: cache guild memberships (refreshed on login + periodically).
CREATE TABLE user_guilds (
    user_id      BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    guild_id     TEXT NOT NULL,
    guild_name   TEXT,
    guild_icon   TEXT,
    refreshed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, guild_id)
);
CREATE INDEX idx_user_guilds_guild ON user_guilds(guild_id);
```

### 4.2 Material

```sql
CREATE TABLE materials (
    id            BIGSERIAL PRIMARY KEY,
    -- Audit trail only — the first user who caused this row to be
    -- inserted. NOT an ownership boundary; subsequent users who
    -- import the same source-backed manga share this row.
    imported_by   BIGINT REFERENCES users(id) ON DELETE SET NULL,
    origin        TEXT NOT NULL CHECK (origin IN ('source','extension','upload')),

    -- Source-backed identity. NULL for ext + upload.
    source        TEXT,        -- 'happymh' | 'mangadex' | …
    upstream_ref  TEXT,        -- mangaUrl as the manifest runtime knows it

    -- Display snapshot — denormalized; refresh when manifest reports
    -- a change. The first user who imported wins the snapshot;
    -- subsequent imports MAY trigger a background refresh but never
    -- mutate ownership.
    title         TEXT NOT NULL,
    cover_url     TEXT,
    description   TEXT,
    author        TEXT,
    status        TEXT,
    languages     TEXT[] NOT NULL DEFAULT '{}',

    -- Identity hints used by community-voted cross-source linking.
    -- Populated by the manifest runtime when the source exposes
    -- them; left NULL otherwise.
    title_native  TEXT,                  -- Japanese / Romaji / native title
    title_alt     TEXT[],                -- aliases the source ships
    cross_refs    JSONB,                 -- e.g. {"mdex_uuid":"…","anilist":12345}

    -- NSFW gate. Set by manifest (manifest.nsfw), by ext UI, or by
    -- user during upload. Forces draft visibility='private'.
    nsfw          BOOLEAN NOT NULL DEFAULT FALSE,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Source-backed materials dedupe across users on (source, upstream_ref).
-- Two users opening HappyMH/Naruto-slug share this single row.
-- Extension / upload materials have source IS NULL → no dedup constraint;
-- each capture / upload gets its own row.
CREATE UNIQUE INDEX uniq_materials_source_ref
    ON materials (source, upstream_ref)
    WHERE source IS NOT NULL;

CREATE INDEX idx_materials_imported_by ON materials(imported_by);

-- Identity-resolution support: find candidates by native title or
-- cross-ref when no explicit user link exists yet.
CREATE INDEX idx_materials_title_native
    ON materials(title_native)
    WHERE title_native IS NOT NULL;
CREATE INDEX idx_materials_cross_refs
    ON materials USING GIN (cross_refs)
    WHERE cross_refs IS NOT NULL;
```

#### 4.2.1 Cross-source linking (community-voted)

```sql
-- One vote per (user, pair). Canonical ordering on the pair
-- (material_a_id < material_b_id) avoids storing A↔B twice.
CREATE TABLE material_link_votes (
    material_a_id  BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    material_b_id  BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    voter_id       BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    vote           SMALLINT NOT NULL CHECK (vote IN (-1, 1)),
    voted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (material_a_id, material_b_id, voter_id),
    CHECK (material_a_id < material_b_id)
);
CREATE INDEX idx_link_votes_a ON material_link_votes(material_a_id);
CREATE INDEX idx_link_votes_b ON material_link_votes(material_b_id);

-- Aggregate score, refreshed periodically (every 5 min) since vote
-- changes are infrequent and a stale read is fine.
CREATE MATERIALIZED VIEW material_links AS
SELECT material_a_id,
       material_b_id,
       SUM(vote)::INTEGER  AS score,
       COUNT(*)::INTEGER   AS total_votes
FROM material_link_votes
GROUP BY material_a_id, material_b_id;
CREATE UNIQUE INDEX uniq_material_links
    ON material_links(material_a_id, material_b_id);
CREATE INDEX idx_links_a ON material_links(material_a_id);
CREATE INDEX idx_links_b ON material_links(material_b_id);
```

Vote semantics:

| User action                                | Effect on votes        |
|--------------------------------------------|------------------------|
| Confirms "yes, this is the same manga"     | +1                     |
| Rejects "no, different manga"              | -1                     |
| Unlinks an entry the user previously linked| Vote row deleted (= 0) |
| Manually links via search                  | +1                     |

Suggestion thresholds (used by `/api/library/suggest`):

```
score ≥ 3   → high confidence, may auto-link silently for new opens
1 ≤ score ≤ 2  → suggest with explicit "Gộp / Bỏ qua" modal
score = 0   → no suggestion via voting (fall back to manifest hints)
score < 0   → never suggest (community has flagged the pair as wrong)
```

Anti-abuse phase 1:

- 50 votes / user / day (`material_link_votes` rate-limited at API).
- Vote score is visible to users (transparency).
- Reputation weighting deferred to phase 2.

### 4.3 Chapter

```sql
CREATE TABLE chapters (
    id                BIGSERIAL PRIMARY KEY,
    material_id       BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,

    position          INTEGER NOT NULL,    -- sparse server-managed sort key
    number            TEXT NOT NULL,       -- display, free-form
    label             TEXT,                -- full label as the source presents
    upstream_url      TEXT,                -- chapter URL on source; NULL for upload

    pages_origin      TEXT NOT NULL CHECK (pages_origin IN ('remote','local')),

    -- Content-addressable storage for prepared.bnl. SHA256 of the
    -- packed archive bytes. Lets two chapters with identical pixel
    -- content (same upload zip, same source content) share prepared
    -- output and downstream caches.
    prepared_hash     TEXT,                 -- hex sha256, NULL until prepare done
    prepared_backend  TEXT,
    prepared_locator  TEXT,
    page_count        INTEGER NOT NULL DEFAULT 0,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (material_id, position)
);
CREATE INDEX idx_chapters_material  ON chapters(material_id, position);
CREATE INDEX idx_chapters_prepared  ON chapters(prepared_hash)
    WHERE prepared_hash IS NOT NULL;
```

`prepared_hash` is the CAS key. Two chapter rows with the same hash
share the same prepared.bnl blob in storage and the same downstream
scan output.

### 4.4 Scan output (Layer 1 — chapter level)

Bubbles, geometry, and masks bind to **chapter_id**, not translation.
They depend only on pixels, so every translation on the chapter reads
the same data.

```sql
CREATE TABLE bubbles (
    chapter_id    BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index    INTEGER NOT NULL,
    bubble_idx    INTEGER NOT NULL,
    source_text   TEXT NOT NULL,
    confidence    REAL NOT NULL,
    shape_kind    TEXT NOT NULL DEFAULT 'dialogue'
                     CHECK (shape_kind IN ('dialogue','burst')),
    source_text_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('simple', source_text)) STORED,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);
CREATE INDEX idx_bubbles_source_text_tsv ON bubbles USING GIN (source_text_tsv);

CREATE TABLE bubble_geometry (
    chapter_id  BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index  INTEGER NOT NULL,
    bubble_idx  INTEGER NOT NULL,
    polygon     JSONB NOT NULL,
    fit_box     JSONB NOT NULL,
    erase_box   JSONB NOT NULL,
    text_box    JSONB NOT NULL,
    PRIMARY KEY (chapter_id, page_index, bubble_idx)
);

CREATE TABLE page_geometry (
    chapter_id  BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    page_index  INTEGER NOT NULL,
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    PRIMARY KEY (chapter_id, page_index)
);

-- masks.npz lives in blob storage; locator on chapter:
ALTER TABLE chapters
    ADD COLUMN masks_backend TEXT,
    ADD COLUMN masks_locator TEXT;
```

### 4.5 Draft (Layer 2 — lang + glossary level, cross-user)

```sql
CREATE TABLE translation_drafts (
    id                 BIGSERIAL PRIMARY KEY,
    chapter_id         BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    source_lang        TEXT NOT NULL,
    target_lang        TEXT NOT NULL,
    glossary_fp        TEXT NOT NULL,    -- hash of applied glossary terms

    llm_model          TEXT NOT NULL,
    created_by         BIGINT NOT NULL REFERENCES users(id) ON DELETE SET NULL,

    -- Auto-share opt-out gate.
    visibility         TEXT NOT NULL DEFAULT 'guild'
                          CHECK (visibility IN ('private','guild','all_guilds')),
    -- For visibility='guild', which guild this draft is scoped to.
    -- Captured from the user's current activity instance at spawn time.
    scope_guild_id     TEXT,

    -- DMCA marker. Set when an admin takes down; cascades to dependent
    -- translations.
    takedown_at        TIMESTAMPTZ,
    takedown_reason    TEXT,

    state              TEXT NOT NULL DEFAULT 'pending'
                          CHECK (state IN ('pending','running','done','error')),
    progress_stage     TEXT,
    progress_index     INTEGER,
    progress_total     INTEGER,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- One draft per cache key. Visibility-private drafts skip the unique
-- constraint (each user gets their own).
CREATE UNIQUE INDEX uniq_drafts_cache
    ON translation_drafts (chapter_id, source_lang, target_lang, glossary_fp)
    WHERE visibility != 'private' AND takedown_at IS NULL;

CREATE INDEX idx_drafts_creator   ON translation_drafts(created_by);
CREATE INDEX idx_drafts_chapter   ON translation_drafts(chapter_id);

CREATE TABLE translation_draft_bubbles (
    draft_id        BIGINT NOT NULL REFERENCES translation_drafts(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    translated_text TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('dialogue','sfx','skip')),
    translated_text_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('simple', translated_text)) STORED,
    PRIMARY KEY (draft_id, page_index, bubble_idx)
);
CREATE INDEX idx_draft_bubbles_text_tsv
    ON translation_draft_bubbles USING GIN (translated_text_tsv);

-- Per-draft LLM context brief (replaces old chapter_briefs keyed by
-- chapter_id).
CREATE TABLE draft_briefs (
    draft_id    BIGINT PRIMARY KEY REFERENCES translation_drafts(id) ON DELETE CASCADE,
    brief_json  JSONB NOT NULL,
    summary     TEXT,
    terms_text  TEXT,
    facts_text  TEXT,
    rules_text  TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_tsv  tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(summary,    '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(terms_text, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(facts_text, '')), 'C') ||
        setweight(to_tsvector('simple', coalesce(rules_text, '')), 'D')
    ) STORED
);
CREATE INDEX idx_draft_briefs_search_tsv ON draft_briefs USING GIN (search_tsv);
```

### 4.6 Translation (Layer 3 — per-user wrapper)

```sql
CREATE TABLE translations (
    id                 BIGSERIAL PRIMARY KEY,
    chapter_id         BIGINT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
    owner_id           BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_lang        TEXT NOT NULL,
    -- Pointer to the cached LLM output. NULL if owner is force-spawning
    -- (private) and the draft hasn't been created yet. After pipeline
    -- finishes, always non-null.
    draft_id           BIGINT REFERENCES translation_drafts(id) ON DELETE SET NULL,

    -- Render archive. If owner has zero edits, shares the draft's
    -- default archive (archive_locator may be NULL — read draft's
    -- default at serve time). If edits exist, archive is rendered
    -- per-translation.
    archive_backend    TEXT,
    archive_locator    TEXT,
    rendered_at        TIMESTAMPTZ,

    -- Hội Mê Truyện feed inclusion. Independent from draft.visibility:
    -- draft can be 'guild'-cached without translation being in the
    -- feed; conversely a private draft can have an in-feed translation
    -- if the owner shares the rendered output explicitly.
    in_feed            BOOLEAN NOT NULL DEFAULT TRUE,
    feed_guild_id      TEXT,            -- which guild's feed

    takedown_at        TIMESTAMPTZ,
    takedown_reason    TEXT,

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (chapter_id, owner_id, target_lang)
);
CREATE INDEX idx_translations_owner    ON translations(owner_id);
CREATE INDEX idx_translations_feed
    ON translations(feed_guild_id, created_at DESC)
    WHERE in_feed = TRUE AND takedown_at IS NULL;

-- Sparse edits over the shared draft. Only edited bubbles get rows.
-- Reader does: load draft bubbles → overlay edits.
CREATE TABLE translation_edits (
    translation_id  BIGINT NOT NULL REFERENCES translations(id) ON DELETE CASCADE,
    page_index      INTEGER NOT NULL,
    bubble_idx      INTEGER NOT NULL,
    edited_text     TEXT NOT NULL,
    edited_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (translation_id, page_index, bubble_idx)
);
```

### 4.7 Glossary

```sql
-- Per-user glossary (replaces old per-project).
CREATE TABLE user_glossary (
    id           BIGSERIAL PRIMARY KEY,
    owner_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    notes        TEXT,
    source_term_tsv tsvector GENERATED ALWAYS AS
        (to_tsvector('simple', source_term)) STORED,
    UNIQUE (owner_id, source_lang, target_lang, source_term)
);
CREATE INDEX idx_user_glossary_text ON user_glossary USING GIN (source_term_tsv);

-- Community glossary scaffold. Phase 1: schema only, seeded by admin.
-- Phase 2: vote + contribution UI.
CREATE TABLE community_glossary (
    id           BIGSERIAL PRIMARY KEY,
    source_lang  TEXT NOT NULL,
    target_lang  TEXT NOT NULL,
    source_term  TEXT NOT NULL,
    target_term  TEXT NOT NULL,
    -- NULL = global term; non-null = scoped to a material.
    material_id  BIGINT REFERENCES materials(id) ON DELETE CASCADE,
    vote_score   INTEGER NOT NULL DEFAULT 0,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_lang, target_lang, source_term, material_id)
);
```

Effective glossary for a user/material:

```
community (scoped to material)  ←  base
  + community (global, same lang pair)
  + user_glossary (overrides)
```

`glossary_fp` for cache key:

```python
def glossary_fingerprint(user_id, source_lang, target_lang, material_id):
    rows = effective_glossary(user_id, source_lang, target_lang, material_id)
    sig = "|".join(sorted(f"{r.source_term}={r.target_term}" for r in rows))
    return sha256(sig.encode()).hexdigest()[:16]
```

Default user with no overrides → community-only fingerprint → high
cache hit rate across users.

### 4.8 Library + bookmark

```sql
CREATE TABLE library_entries (
    id                  BIGSERIAL PRIMARY KEY,
    user_id             BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title               TEXT NOT NULL,
    cover_url           TEXT,
    primary_material_id BIGINT REFERENCES materials(id) ON DELETE SET NULL,
    bookmarked          BOOLEAN NOT NULL DEFAULT FALSE,
    bookmarked_at       TIMESTAMPTZ,
    last_read_at        TIMESTAMPTZ,
    last_chapter_ref    JSONB,  -- {material_id, chapter_id, label}
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_library_user ON library_entries(user_id);

CREATE TABLE library_materials (
    entry_id     BIGINT NOT NULL REFERENCES library_entries(id) ON DELETE CASCADE,
    material_id  BIGINT NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
    link_origin  TEXT NOT NULL CHECK (link_origin IN ('primary','auto','manual')),
    linked_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entry_id, material_id)
);
CREATE UNIQUE INDEX uniq_library_material_per_user
    ON library_materials (material_id);
-- Combined with the FK chain, a material appears in at most one
-- library_entry per user.
```

Bookmark moves to `library_entries.bookmarked` — no separate
`material_pins` table.

### 4.9 Pipeline coordination

```sql
-- Re-keyed. Prepare and scan key by chapter_id (chapter-level work).
-- Translate keys by draft_id. Render keys by translation_id (or
-- draft_id for the default shared render).
CREATE TABLE tasks (
    target_kind   TEXT NOT NULL CHECK (target_kind IN ('chapter','draft','translation')),
    target_id     BIGINT NOT NULL,
    stage         TEXT NOT NULL CHECK (stage IN ('prepare','scan','translate','render')),
    claimed_by    TEXT,
    claimed_at    TIMESTAMPTZ,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    PRIMARY KEY (target_kind, target_id, stage)
);
CREATE INDEX idx_tasks_claim ON tasks(stage, claimed_by, claimed_at);

-- (LISTEN/NOTIFY trigger same as before, payload now
-- '<target_kind>:<target_id>'.)
```

### 4.10 Quota + DMCA

```sql
-- One row per LLM-costing event (cache miss draft spawn or fresh render).
-- Cache HITS do NOT insert. Quota counts what the user actually paid for.
CREATE TABLE chapter_consumes (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    translation_id  BIGINT REFERENCES translations(id) ON DELETE SET NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('draft_create','render_create')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_chapter_consumes_user_time
    ON chapter_consumes(user_id, created_at DESC);

CREATE TABLE dmca_takedowns (
    id              BIGSERIAL PRIMARY KEY,
    target_kind     TEXT NOT NULL CHECK (target_kind IN
                       ('material','chapter','draft','translation')),
    target_id       BIGINT NOT NULL,
    scope_guild_id  TEXT,
    reason          TEXT NOT NULL,
    reporter        TEXT NOT NULL,
    taken_down_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    restored_at     TIMESTAMPTZ
);
```

## 5. Storage layout

```
c/{chapter_id}/prepared.bnl              Chapter-level, deduped via CAS hash
c/{chapter_id}/masks.npz                 Chapter-level
d/{draft_id}/render.bnl                  Default render archive (shared across
                                         translations that reference this draft
                                         and have no edits)
t/{translation_id}/render.bnl            Per-translation when sparse edits exist
                                         (forks the draft render)
```

Public URL pattern unchanged:

```
https://{da-host}/cdn/t/render/{HMAC(target_kind:target_id, salt)}.bnl?v={updated_at}
```

`HMAC` input is `"draft:42"` or `"translation:42"`. Knowledge of URL
implies capability — auth gate is the frontend, which only emits URLs
for entities the user has permission to read.

CDN edge cache (Discord proxy) handles repeat reads of hot archives
for free.

## 6. Pipeline — translation flow detail

### 6.1 Spawn endpoint contract

```
POST /api/translate
Body:  { chapter_id, target_lang, force_private?: bool }
Auth:  Discord JWT
       (frontend has resolved chapter_id via /api/material/{...} flow)

Behavior:
  1. Resolve user_id, current scope_guild_id (from JWT claims).
  2. Resolve effective glossary → glossary_fp.
  3. Lookup draft (chapter_id, source_lang, target_lang, glossary_fp,
     visibility != 'private', NOT taken_down).
  4. If hit AND user can_use_draft(draft):
       Reuse draft.
       Create translation row (chapter_id, user, target_lang, draft_id).
       Return 200 { translation_id, state, cache_hit: true }.
       Quota: 0 (no chapter_consumes row).
  5. Else (miss OR private OR not authorized):
       Create draft (visibility=force_private ? 'private' : 'guild',
                     scope_guild_id=current_guild,
                     creator=user).
       Create translation row (draft_id=new draft).
       Enqueue tasks: ensure prepare done → scan → translate → render.
       Insert chapter_consumes (kind='draft_create').
       Return 200 { translation_id, draft_id, state: 'pending',
                    cache_hit: false }.
```

### 6.2 Stage execution

```
Stage 1: prepare
  Target: chapter_id
  Skip if chapter.prepared_hash IS NOT NULL.
  Otherwise: pack pages → hash → write c/{id}/prepared.bnl.
  Update chapter.prepared_hash, prepared_locator.
  Next: enqueue scan if any draft pending on this chapter.

Stage 2: scan
  Target: chapter_id
  Skip if bubbles rows exist for chapter.
  Otherwise: OCR + geometry → write bubbles, bubble_geometry,
             page_geometry, masks.npz.
  Next: enqueue translate for every pending draft on this chapter.

Stage 3: translate
  Target: draft_id
  Input: bubbles + briefs + effective glossary.
  Call LLM with target_lang.
  Write translation_draft_bubbles + draft_briefs.
  Update draft.state='done'.
  Next: enqueue render for every translation linked to this draft.

Stage 4: render
  Target: translation_id (default) OR draft_id (shared-archive case).
  If translation has no edits:
    Render once per draft (target_kind='draft'); skip if
    d/{draft_id}/render.bnl exists. All translations reference this.
  If translation has edits:
    Render per translation (target_kind='translation') → t/{id}/render.bnl.
  Update archive_* on translation row.
```

### 6.3 SSE events

```
GET /api/translate/{translation_id}/events     SSE

Events emitted:
  draft.progress     { stage, index, total }
  draft.done         { draft_id }
  render.progress    { index, total }
  render.done        { archive_url }
  error              { stage, message }
```

When the translation reuses a cached draft (cache_hit), the SSE
stream emits `draft.done` immediately followed by render flow (or
`render.done` immediately if the draft's default render is already
available).

### 6.4 Failure cases

| Case | Behavior |
|---|---|
| Draft hits cache but takedown_at set | Treat as miss; spawn new draft (different glossary_fp won't help since takedown is per-draft, not per-key; user spawns a fresh private draft) |
| User force_private but cache exists | Always honor `force_private` — spawn fresh private draft, no reuse |
| Glossary mismatch (user has overrides) | New glossary_fp → cache miss → spawn fresh draft. Cost is on the user with overrides, not the community |
| NSFW material | force_private auto-set; draft.visibility='private' regardless of opt-out |
| User leaves guild between spawn and read | Visibility check at read time uses current guild memberships. User may lose access to a draft they linked to; system falls back to spawning a private draft on next read |

## 7. API surface

### 7.1 Auth

```
POST /api/auth/discord/exchange    Exchange Discord OAuth code → JWT
GET  /api/me                       Current user + guilds
                                   Refreshes user_guilds cache
```

No password, no email, no API token for users (extension still uses
API tokens — see RFC-009).

### 7.2 Material

```
POST /api/material/import          { source, upstream_ref } → ensure row,
                                   returns material_id. Idempotent per
                                   (owner, source, upstream_ref).
POST /api/material/upload-init     Multipart upload init (unchanged shape)
POST /api/material/upload-finalize Multipart upload finalize → material_id
GET  /api/material/{id}            Material + chapters with translation overlay
PATCH /api/material/{id}           title, cover, nsfw (owner only)
DELETE /api/material/{id}          Cascades to chapters, translations, blobs
```

Chapter list overlay:

```jsonc
{
  "id": 4711,
  "number": "1099",
  "translations": [
    { "id": 9001, "creator": "@nghyane", "lang": "vi",
      "state": "done", "from_cache": true, "in_feed": true },
    { "id": 9003, "creator": "@user2",  "lang": "en",
      "state": "running", "from_cache": false, "in_feed": true }
  ]
}
```

Frontend can render the chapter row directly. No second round-trip.

### 7.3 Translate

```
POST  /api/translate                       see §6.1
GET   /api/translate/{id}                  detail
POST  /api/translate/{id}/redo             re-run (force fresh draft)
PATCH /api/translate/{id}                  in_feed, force_private
DELETE /api/translate/{id}                 cascades archive cleanup
SSE   /api/translate/{id}/events           §6.3
```

### 7.4 Library

```
GET  /api/library                          entries (with linked materials)
POST /api/library/entry                    create entry (from a material)
PATCH /api/library/entry/{id}              bookmark, title override
DELETE /api/library/entry/{id}             also unlinks materials
POST /api/library/entry/{id}/link          link material → also casts +1 vote
                                           on the (existing_material, new) pair
POST /api/library/entry/{id}/unlink        unlink material → removes the vote
                                           (drops entry if 0 materials left)
GET  /api/library/suggest?material_id=...  see §7.4.1
POST /api/library/suggest/{candidate_id}/reject
                                           explicit -1 vote on the suggested
                                           pair (user clicks "Không, manga khác")
```

#### 7.4.1 Suggestion ranking (server-side)

The suggest endpoint blends three signals in priority order. The
first one to clear its threshold wins:

```
1. cross_refs match  — material.cross_refs ∋ key X
                        AND another material in user's library has
                        cross_refs ∋ same X  (e.g. shared mdex_uuid)
                       → high confidence, auto-link if vote score ≥ 0

2. community votes   — material_links.score ≥ 3 for some pair
                        (this_material, other) where other is in
                        user's library
                       → high confidence, present "Gộp luôn?"

3. title_native      — case-folded equality between this material's
                        title_native and any in user's library
                       → medium confidence, present "Gộp / Khác manga"

4. community votes   — score in [1, 2]
                       → low confidence, present "Có thể là cùng manga?"

5. nothing fires     → no suggestion; user must use manual search
```

Manifest hints (signals 1 & 3) are the **cold-start fuel**. Voting
(signals 2 & 4) takes over as the system accumulates user actions.

### 7.5 Feed (Hội Mê Truyện, guild-scoped)

```
GET /api/feed/guild/{guild_id}            Latest translations in_feed=TRUE
                                          scoped to guild. User must be member.
```

No `/api/feed/public`. No global feed. Member-only.

### 7.6 DMCA

```
POST /api/dmca/report                     User-facing report form
GET  /api/admin/dmca                      Admin list (admin scope only)
POST /api/admin/dmca/{id}/takedown        Mark target taken_down_at
POST /api/admin/dmca/{id}/restore         Reverse
```

## 8. Migration plan

### 8.1 Branching

```
main
  └─ checkpoint f8977dd                ← rollback point
       └─ feat/material-architect      ← integration branch
            ├─ feat/material-schema
            ├─ feat/material-api
            ├─ feat/material-pipeline
            ├─ feat/material-storage
            ├─ feat/material-web
            └─ feat/material-ext
```

Sub-branches merge into `feat/material-architect`. Integration
branch merges to `main` as one commit. Rollback = `git revert <merge>`
or `git reset --hard f8977dd`.

### 8.2 Data migration

**Decision: drop & recreate**. Beta user count is small; preserving
state isn't worth the migration tooling cost. Operator runs:

```
dropdb typoon
createdb -O typoon typoon
psql typoon < typoon/storage/schema.sql
```

Pre-flag-day, take a snapshot for emergency rollback:

```
pg_dump --format=custom typoon > checkpoint-$(date +%s).dump
```

### 8.3 URL compatibility

```
Old → New:
  /projects/$id                            → no redirect (URLs were behind auth, not bookmarked externally)
  /projects/$id/chapters/$cid              → no redirect
  /browse/$source/manga/$id                → 301 → /manga/$source/$id
  /browse/$source/manga/$id/chapter/$cid   → 301 → /manga/$source/$id/chapter/$cid
```

The `/browse/...` redirects are cheap (TanStack Router static map);
preserve in-app bookmarks during the beta. Project URLs were never
bookmarked outside DA, so we don't redirect them.

### 8.4 CDN cache

`archive_token` input changes from `(project_id, chapter_id)` to
`(target_kind, target_id)`. Every cached URL is stale. Rotate
`BLOB_SALT` on flag day — all existing tokens become invalid;
clients receive 404 on next fetch, frontend re-resolves URL via API,
get new token, archive re-served.

### 8.5 Extension migration

Extension currently uses project-based flow. Rewrite to:

```
1. User on a manga page → ext detects source + manga URL.
2. POST /api/material/import { source, upstream_ref } → material_id.
3. Ext capture chapter pages.
4. POST /api/material/upload-init|finalize attached to material_id +
   new chapter row.
5. Optionally POST /api/translate (auto-spawn with user's default
   target_lang).
```

The "create project" picker disappears. Materials auto-resolve.
Ships **in the same PR** as the web rewrite — no parallel paths.

### 8.6 Flag day

1. Code-freeze `feat/material-architect`.
2. `pg_dump` production DB (rollback insurance).
3. Stop `com.typoon.api` + `com.typoon.worker`.
4. `dropdb && createdb && psql < schema.sql` on production DB.
5. Deploy new backend (git fetch + restart LaunchAgents).
6. Rotate `BLOB_SALT` in `.env`.
7. `wrangler pages deploy` new SPA.
8. Smoke test (see deploy-beta.md §7 + new probes).
9. Tag merge SHA in this RFC: "shipped at `<sha>`".

Rollback procedure:

```
launchctl unload ~/Library/LaunchAgents/com.typoon.{api,worker}.plist
git reset --hard f8977dd
dropdb typoon && createdb -O typoon typoon
pg_restore -d typoon checkpoint-<ts>.dump
launchctl load ~/Library/LaunchAgents/com.typoon.{api,worker}.plist
wrangler pages deploy web/dist  # using web/dist built from f8977dd
```

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Pipeline regression sharing prepare/scan | M | H | E2E test: 2 users spawn same chapter, both succeed, second is cache_hit=true |
| Visibility bypass leaks private draft | L | H | Authorization check at READ time, not just spawn; covered by integration tests |
| User loses guild between spawn and read | M | M | Visibility falls back to private clone draft; doesn't break, just re-spawns |
| CDN cache stale → 404 storm | M | L | Salt rotation kills cleanly; first fetch after flip re-resolves URL |
| Extension behind on rewrite | M | M | Ext is in `ext/` subtree; can disable popup button until ext ships; document in CHANGELOG |
| Glossary collapse merges divergent project glossaries | L | M | One-shot SQL UNION at migration; document; owners with multiple projects on same lang pair get merged glossary they can prune |
| DMCA takedown notice while admin offline | L | H | Disable in_feed for reported translation pending admin review (15-min auto-mute on report); proper takedown when admin handles |
| Guild ID expires (Discord guild deleted) | L | L | scope_guild_id becomes orphan; drafts still readable by creator; feed entries 404 cleanly |
| LLM cost spike from bug spawning duplicate drafts | M | H | Unique index on (chapter, src, tgt, glossary_fp) where not private + not takedown prevents duplicate non-private drafts |

## 10. What stays the same

- Discord OAuth, JWT minting, API tokens (RFC-008/009).
- Render archive Bunle format (`.bnl`).
- Worker LISTEN/NOTIFY pattern; payload format changes.
- Frontend manifest runtime (`features/browse/manifest/*`) —
  material `import` calls the runtime exactly as today.
- Browse hub, source picker, shelves, search, infinite scroll.
- Library local-first store (Phase A) — store shape expands to
  track material_id alongside (source, ref); bookmark writes
  through to `/api/library/entry/{id}`.
- DA setup, deploy topology, CORS, TRUSTED_HOSTS.

## 11. What gets deleted

```
typoon/api/routes/projects.py             → split into material.py
                                            + chapter.py + translate.py
typoon/api/routes/project_events.py       → translate_events.py
typoon/adapters/projects.py               → split: material.py + translate.py

web/src/features/project-detail/          → merged into features/manga/
web/src/routes/projects.*.tsx             → deleted; replaced by /manga/$source/$ref
web/src/features/library/internal.ts*     → no longer needed (no community source)
```

Wiki pages updated:

```
docs/wiki/architecture.md                 — package layout
docs/wiki/browse-mode.md                  — §1/§5/§10/§11 (material model)
docs/wiki/render-archive-storage.md       — key layout, salt rotation
docs/wiki/deploy-beta.md                  — probe list, flag-day playbook
docs/wiki/hard-rules.md                   — strike project pipeline
docs/wiki/material-architecture.md        — promoted to canonical (no longer draft)
```

## 12. Out of scope (Phase C+)

- Chapter reader unification (internal `.bnl` reader vs external raw
  reader). Same hero, different page loader. Phase C.
- Translation rating / claim ("I want to translate this").
- Per-chapter glossary override (translation-level glossary).
- Watch Party shared-scroll session.
- Discord rich presence ("Reading X · chapter Y").
- Background new-chapter refresh worker.
- Cross-device history sync (currently local-only).
- External ID matching (MangaDex universal ID).
- Per-page CAS (page-pixel hash for cross-chapter dedup).
- Fine-grained glossary fingerprint (applied-terms intersection).

## 13. Decisions (confirmed)

| Question | Decision |
|---|---|
| Drop or migrate data | Drop & recreate (§8.2) |
| Translation share visibility | Per-translation `in_feed` flag, decoupled from draft visibility |
| Material identity for source-backed | **Cross-user dedup** on `(source, upstream_ref)` (v5 reversal of v4 per-user). `imported_by` is audit, not ownership. |
| Material identity for upload / ext | Per-row (no dedup) — no upstream_ref to dedup against |
| Cross-source linking strategy | **Community voting** on `material_link_votes` + manifest identity hints (`title_native`, `cross_refs`). No external MangaDex API call. |
| Cold start for voting | Admin seeds top-100 cross-source pairs on launch; manifest hints carry long tail; voting accrues organic |
| Auto-spawn translation on ext capture | Yes — ext POSTs `/translate` with user's default target_lang |
| Synthetic `source='community'` | Gone — Hội Mê Truyện is a feed query, not a source |
| Auto-share opt-out | Default checked at spawn modal; uncheck → `force_private=true` |
| Cache key shape | `(chapter_id, source_lang, target_lang, glossary_fp)` for non-private drafts |
| CAS for prepared.bnl | Yes — `chapter.prepared_hash` SHA256 |
| Sharing scope | Guild-scoped (`visibility ∈ {private,guild,all_guilds}`) |
| Public web access | Removed — DA-only deployment |
| Default guild for new users | "Hội Mê Truyện Official" Discord guild — bot auto-invites on first login |
| Manifest schema extension | Add `title_native`, `title_alt`, `cross_refs` selector slots; sources expose what they have (HappyMH `data-mdex-id`, MangaDex native UUID, OTruyen `english_name`) |

Shipped at: _<pending merge SHA>_
