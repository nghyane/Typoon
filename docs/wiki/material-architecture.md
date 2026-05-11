# Material architecture — at a glance

> **Status**: planning, final design. See [`docs/rfc/material-architecture.md`](../rfc/material-architecture.md)
> for the full RFC. Implementation begins from `f8977dd`.
>
> This card is what you grep when writing code on `feat/material-architect`.

## Entities (3-layer cache + per-user wrapper)

```
Material         Per-user manga identity (any origin)
  Chapter        Pages, CAS-deduped via prepared_hash
    Bubbles, Geometry, Masks   ← chapter-level (Layer 1, shared)
    
    TranslationDraft           ← (chapter, src, tgt, glossary_fp) cache key
                                 visibility ∈ {private, guild, all_guilds}
      DraftBubbles (text)      ← Layer 2 (shared by visibility scope)
      DraftBrief (LLM context)
      
      Translation              ← per-user wrapper
        in_feed flag           ← Layer 3 (per user)
        sparse Edits override
        render archive (shared with draft if no edits)

LibraryEntry     Per-user grouping of Materials representing same manga
```

## Sharing scope

DA-only: no public web. Sharing is **guild-scoped**.

```
visibility = 'private'      — owner only
visibility = 'guild'        — members of scope_guild_id
visibility = 'all_guilds'   — members of any guild creator is in
```

Hội Mê Truyện = `GET /api/feed/guild/{guild_id}` filter
`in_feed=TRUE AND takedown_at IS NULL`. No global feed.

## Cache flow (one diagram)

```
User clicks "Dịch chỉn chu" on chapter row
                 ↓
        POST /api/translate
                 ↓
   ┌───────────────────────────────────────┐
   │ Lookup draft by                        │
   │   (chapter_id, src, tgt, glossary_fp)  │
   │   WHERE visibility != 'private'        │
   │   AND can_use(user, draft)             │
   └────────────┬──────────────────────────┘
                ↓                
        ┌───────┴───────┐
        ↓               ↓
    CACHE HIT       CACHE MISS
   create translation     create draft (visibility=guild,
   row → draft_id           scope=user's current guild)
   archive=draft default  create translation row
   quota = 0              enqueue prepare → scan → translate → render
   ~30ms                  quota = 1 (chapter_consumes)
                           ~2 minutes
```

## Pipeline stages + ownership

| Stage | Keyed by | Skips if | Cost |
|---|---|---|---|
| prepare | `chapter_id` | `chapter.prepared_hash IS NOT NULL` | CPU 5s |
| scan | `chapter_id` | bubbles exist for chapter | GPU 1-3 min |
| translate | `draft_id` | draft_bubbles exist for draft | LLM $0.03-0.08 |
| render | `draft_id` or `translation_id` | archive exists (with/without edits) | CPU 30s |

## Storage keys

```
c/{chapter_id}/prepared.bnl       chapter-level, CAS-deduped
c/{chapter_id}/masks.npz          chapter-level
d/{draft_id}/render.bnl           default render (no-edits case)
t/{translation_id}/render.bnl     per-translation when edits exist
```

Public URL = `https://{da-host}/cdn/t/render/{HMAC(target_kind:id, salt)}.bnl?v=...`

## Routes (final)

```
Discovery & reading:
  /browse                              source hub
  /browse/$source                      source landing
  /browse/community                    list of user's guilds with activity
  /browse/community/$guild_id          Hội Mê Truyện feed for one guild
  /manga/$source/$ref                  material detail (any origin)
  /manga/$source/$ref/chapter/$id      reader (Phase C unifies)

Personal:
  /library                             cross-source library
  /translate                           my translations list
  /translate/$id                       editor view
```

No `/projects/*`. Ever.

## API (final)

```
Auth:
  POST  /api/auth/discord/exchange       Discord OAuth code → JWT
  GET   /api/me                          User + guilds

Material:
  POST  /api/material/import             {source, upstream_ref} → row
  POST  /api/material/upload-init|finalize
  GET   /api/material/{id}               + chapters + translations overlay
  PATCH /api/material/{id}
  DELETE /api/material/{id}

Translate:
  POST  /api/translate                   {chapter_id, target_lang, force_private?}
  GET   /api/translate/{id}
  POST  /api/translate/{id}/redo
  PATCH /api/translate/{id}              in_feed, force_private
  DELETE /api/translate/{id}
  SSE   /api/translate/{id}/events

Library:
  GET   /api/library                     entries + linked materials
  POST  /api/library/entry
  PATCH /api/library/entry/{id}          bookmark, title override
  DELETE /api/library/entry/{id}
  POST  /api/library/entry/{id}/link|unlink
  GET   /api/library/suggest?material_id=

Feed:
  GET   /api/feed/guild/{guild_id}       member-only

DMCA:
  POST  /api/dmca/report
  GET   /api/admin/dmca
  POST  /api/admin/dmca/{id}/takedown|restore
```

## Capabilities (UI-derived)

```
read              always (gated by visibility)
bookmark          always (toggles library entry)
spawn_translation always
edit_translation  owner only
mark_in_feed      owner only (default true)
edit_material     imported_by = user (ext/upload only)
delete_material   imported_by = user
admin_takedown    user has admin role in scope guild
```

## Hard rules

- Translation is the work unit. Spawned from chapter rows.
- Material is the reading unit. Library + bookmark attach here.
- Sharing is guild-scoped. No public feed. No SEO.
- Material is per-user. No cross-user dedup at identity level. CAS
  handles compute dedup separately (`prepared_hash`).
- Draft visibility gates cache hit eligibility. Read-time
  authorization, not just spawn-time.
- "Hội Mê Truyện" is a feed query, not a source row.
- Public web access is permanently removed.
- All artifact paths use `{target_kind}:{target_id}` HMAC tokens.

## Migration

- Branch from `f8977dd`.
- Drop & recreate Postgres (beta acceptable wipe).
- Rotate `BLOB_SALT` to invalidate CDN cache.
- Single merge commit. `git revert` or `git reset --hard f8977dd` to
  roll back.
- See [`docs/rfc/material-architecture.md`](../rfc/material-architecture.md) §8.6
  for flag-day playbook.
