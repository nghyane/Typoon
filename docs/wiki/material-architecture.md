# Material architecture — at a glance

> **Status**: planning, final design (v5). See [`docs/rfc/material-architecture.md`](../rfc/material-architecture.md)
> for the full RFC. Implementation begins from `f8977dd`.
>
> This card is what you grep when writing code on `feat/material-architect`.

## Entities (3-layer cache + per-user wrapper)

```
Material         Manga identity, source-agnostic.
  Source-backed: CROSS-USER (unique on (source, upstream_ref)).
                 imported_by is audit, not ownership.
  Ext / upload:  per-row (no dedup).
  
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

LibraryEntry     Per-user grouping of Materials representing same manga.
                 Bookmark, last-read, "Tiếp tục đọc" hang off here.
                 Cross-source links via community voting (passive).
```

## Cross-source identity (community-driven)

No external API (no MangaDex/AniList lookup). Three signals, in
priority order:

```
1. manifest cross_refs match      mdex_uuid / anilist_id from source page
2. community vote score ≥ 3       material_link_votes (auto-suggest)
3. manifest title_native match    Japanese/Romaji from source
4. community vote score 1-2       (suggest with confirmation)
5. nothing                        manual link only
```

Vote semantics:

| User action            | Vote effect           |
|------------------------|-----------------------|
| Confirm "same manga"   | +1                    |
| Reject "different"     | -1                    |
| Manually link          | +1                    |
| Unlink                 | Vote row deleted (=0) |

Anti-abuse phase 1: 50 votes/user/day, score visible, no reputation
weighting yet.

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

## Manifest schema (extended for identity hints)

Three selector slots added so sources can publish what they know:

```jsonc
{
  "endpoints": {
    "manga": {
      "fields": {
        "title":        "...",            // unchanged
        "cover":        "...",            // unchanged
        "title_native": ".info @data-jp", // Japanese / Romaji canonical
        "title_alt":    ".aliases .item", // other-language aliases (array)
        "cross_refs":   "..."             // optional structured ID lookup
      }
    }
  }
}
```

Sources fill in what they ship. HappyMH has `data-mdex-id` on its
detail page; MangaDex's UUID IS its `upstream_ref`; OTruyen has
`english_name`. Each new source costs one extra selector line per
field it can populate.

## API surface (relevant deltas vs RFC §7)

```
POST /api/library/entry/{id}/link              → also casts +1 vote
POST /api/library/entry/{id}/unlink            → removes vote
POST /api/library/suggest/{candidate}/reject   → -1 vote on pair
GET  /api/library/suggest?material_id=         → returns 0-N candidates
                                                  per §7.4.1 ranking
```

## Capabilities (UI-derived)

```
read              always (gated by visibility)
bookmark          always (toggles library entry)
spawn_translation always
edit_translation  owner only
mark_in_feed      owner only (default true)
edit_material     imported_by = user (ext/upload only)
delete_material   imported_by = user (only meaningful for non-source-backed)
admin_takedown    user has admin role in scope guild
```

## Hard rules

- Translation is the work unit. Spawned from chapter rows.
- Material is the reading unit. Library + bookmark attach here.
- Source-backed Material is cross-user. `imported_by` is audit.
- Cross-source linking is **community-voted**, never fuzzy-matched
  on title strings, never cover-hashed, never external-API resolved.
- Sharing is guild-scoped. No public feed. No SEO.
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
- Admin seeds top-100 cross-source manga link votes on launch.
- See [`docs/rfc/material-architecture.md`](../rfc/material-architecture.md) §8.6
  for flag-day playbook.

