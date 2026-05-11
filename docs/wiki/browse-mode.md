# Browse mode — design playbook

Patterns and decisions for `/browse/*` in the web app. Read before
touching any source manifest, view, or route under `features/browse/`.
Skip the file structure (the code is the truth there); this page is
**rules** and **why**.

## 1. Source model — one mental model for any kind

Two kinds of source plug into the same browse UI:

| `kind`     | Examples              | Data backend             | Routing              |
|------------|-----------------------|--------------------------|----------------------|
| `external` | HappyMH, MangaDex, …  | bunle-cdn proxy → site   | `/browse/$src/manga/...` |
| `internal` | Community             | typoon `/api/projects`   | `/projects/{id}`        |

Both are listed at `/browse`, both appear in source picker, both have
shelves and (where supported) search.

**Rule**: views must never branch on `kind` directly. The runtime
exports helpers (`getShelves`, `hasSearch`, `shelfPageSize`,
`fetchBrowse`) that present a unified surface. Branching for routing
targets — `<MangaCard>` deciding `/projects/{id}` vs `/browse/manga/...`
— is the only allowed branch, and it's local to the link element.

Why: keeps shelf rails, search results, pagination logic, infinite
scroll all source-agnostic. Adding a third internal source (Community
v2, dedicated "official" pipeline, …) is one adapter file in
`features/browse/manifest/internal.ts`, no view churn.

## 2. Manifest schema — what each field is for

A source manifest is **declarative**. No code runs from it. The
runtime reads selectors + URL templates, executes them via `pfetch`,
maps to typed result objects.

| Field         | Required when     | Purpose |
|---------------|-------------------|---------|
| `id`, `name`  | always            | Identity + display label |
| `host`        | external          | Proxy allowlist + default Referer |
| `homepage`    | optional          | "Open original" links |
| `languages`   | always            | BCP-47 list, drives lang badge |
| `kind`        | optional (default `external`) | Selects runtime dispatch |
| `accent`      | optional          | Tinted icon backdrop (palette key or hex) |
| `icon`        | optional          | Override monogram with image URL |
| `nsfw`        | optional          | Renders an NSFW tag |
| `endpoints`   | external only     | Shelves + search + manga + chapter |

### Selector grammar (in `manifest/selectors.ts`)

```
$.json.path[*]@key.sub      JSON wildcard + nested key (drop nulls)
$.list[0].key                indexed JSON
@field.subfield              row-relative path
sel || fallback1 || fallback2  pipeline fallback, first non-empty wins
*  / [*]                     wildcard at any depth
```

Trailing wildcard returns **first non-empty value**, not the array.
This is the difference that makes MangaDex `title.en || title.*`
collapse to a string rather than `[string]`.

### Template substitution (in `manifest/runtime.ts: tpl`)

```
{var}     literal substitution (default; for URL path parts)
{var:q}   URL-encoded (for query values that may contain & = space …)
```

Default is **not** encoded because most uses are `{mangaUrl}` / `{code}`
which are already URL-shaped. Only encode `q`, search keywords, etc.

### `extras` + `fields` two-pass

Inside a row, manifest can declare:
- `extras: { code: "@manga_code" }` — pulled first (selectors)
- `fields: { url: "=https://site.com/manga/{code}" }` — templates resolve **after** extras

Order in JSON doesn't matter; runtime sorts: selectors first, then
`=template` fields. Template can reference any other resolved field +
globals (`mangaUrl`, regex named groups from `extract`, …).

## 3. Shelves over tabs

Source landing renders shelves (horizontal rails), not tabs. Decision:

| Tabs (rejected)                       | Shelves (accepted)                   |
|---------------------------------------|---------------------------------------|
| User must understand source taxonomy  | Each shelf has a clear intent        |
| 1 list at a time, requires click      | All shelves visible at once          |
| "Day/Week/Month" tied to source data  | "Phổ biến / Mới cập nhật / Top tuần" — source-agnostic |
| Empty tab = dead end                  | Skeleton rail shows pending          |

Pattern lifted from Apple Music Browse, Steam Discovery, Spotify
"Made For You". Production-tested at scale.

**Shelf id is a URL segment** (`/browse/$src/shelf/$shelfId`).
Renames break bookmarks. Keep stable.

## 4. Three routes only

```
/browse/$src                   landing — shelves feed
/browse/$src/shelf/$shelfId    full grid for one shelf (+ filter bar)
/browse/$src/search?q=...      search query results
/browse/$src/manga/$mangaId    detail (external only)
```

`Filter` lives **only on shelf detail page**, never on landing. Filter
on landing makes the user choose taxonomy before they know what's
there; on detail it refines an already-engaged listing.

Search uses its own route, not a tab — because typing changes intent
fundamentally (browsing → seeking). URL-bound `?q=...` so it's
shareable / back-navigable.

## 5. Library surfaces reading continuity across sources

`features/library/store.ts` — zustand persist (`typoon.library.v1`),
keyed by `(source, mangaUrl)`. One entry carries both bookmark flag
and last-read tracking; the cross-source `/library` route and the
per-source "Tiếp tục đọc" rail render from the same store.

- MangaPage `recordView` on mount (snapshots title/cover + chapters[0]
  as `latestChapter`).
- BrowseReader `markChapterRead` on chapter open (sets `lastReadAt` +
  `lastChapterRead`).
- `hasNewChapter(entry)` derives client-side: `latestChapter.url !==
  lastChapterRead.url` after both have fired at least once.

**Rule**: never mark in MangaCard click — too eager. Mark when the
manga *page* mounts (recordView), and when the reader mounts
(markChapterRead). The store dedupes identity-field writes so
re-renders don't churn localStorage.

`useShallow` is mandatory for any selector that **derives** a list
from the store. Without it, every render produces a new array
reference → zustand triggers re-render → infinite loop. The
"Maximum update depth exceeded" error in `BrowseSourceHome` was this.
See `features/library/hooks.ts` for the wrapped selectors.

**Sub-rule** for `useShallow`: only return *raw store values*; don't
build fresh objects inside the selector. `useShallow` compares
elements with `Object.is`, so `.map(toViewModel)` produces a new
object per item and shallow-compare always fails. Map outside the
selector inside a `useMemo` (see `useUnifiedLibrary` in
`features/library/unified.ts`).

Backend involvement: **zero**. Library refresh happens browser-side
via the DA proxy (free egress per `deploy-beta.md` §5); no worker,
no `/api/library`. Phase 2 may add an opt-in sync for cross-device,
but the local store remains source of truth.

## 6. Auto-translate is a setting, not a UI ornament

Pattern (in `features/browse/autoTranslate.ts`):

- One global toggle persisted to localStorage
- `shouldTranslate(enabled, target, sourceLanguages)` — skip when
  source language matches target (OTruyen → vi == vi)
- Translation deterministic → cache forever (`staleTime: Infinity`)

Google's `translate.googleapis.com/translate_a/single?client=gtx`
endpoint:
- Free, unauth, ~50-150ms latency
- Rate gate 12 req/s client-side
- Batch via `<x>...</x>` HTML wrap (Google preserves tags, translates
  each item in its own context). Sentinels like `@@SEP@@` cause
  cross-item bleed.
- Fallback to original text on any failure. Never error UI.

`<x>` discovery cost a debugging session. Don't change it.

## 7. Proxy header pass-through (bunle-cdn)

`pfetch(url, { headers: { ... } })` from web → `X-Proxy-Headers`
base64url(JSON) → bunle-cdn → upstream.

`proxify(url, headers?)` for `<img src>` uses `?_h=...` query form
because img can't set headers.

**Allow list is gone**; only a denylist of hop-by-hop / infra
headers. Adapters can forward arbitrary Auth/Sign/Referer/UA the
upstream needs. Cache key intentionally excludes request headers
so Cookie/Auth don't leak through edge cache.

Per-host PROFILES was removed. Default = `Referer: https://{host}/`
covers ~90% of hotlink checks. Per-endpoint `manifest.headers`
overrides when needed.

## 8. UI density — Discord-grade

The app lives in Discord Activity. Tune for PIP (360×640) → Expanded
(1920×1080), not for web standards.

| Element            | Web standard | This app    |
|--------------------|--------------|-------------|
| Bar height         | 56-64px      | 32-40px     |
| Input height       | 40px         | 36px        |
| Row height         | 56-72px      | 40-48px     |
| Hero title         | 32-40px      | 24-28px     |
| Gap between rows   | 8-16px       | 4-6px       |

`text-text` / `text-muted` / `text-subtle` form a 3-tier ladder. Don't
add ad-hoc `opacity-N` — use the ladder. `surface < surface-2 <
hover` for elevation; never `border` (it's darker than `bg` and reads
as a black groove).

## 9. Icon system anti-patterns

Things we tried and dropped:

| Tried                          | Why it failed |
|--------------------------------|---------------|
| 2×2 cover collage              | Async fetch, flicker, source identity tied to "hot manga of today" |
| Fetch favicon, scale to tile   | 16-32px → blur unacceptably at 48-56px |
| Saturated palette monogram tile| Tinted color tiles read as toy UI, clash with manga covers |
| Per-source `lucide` icon name  | Manifests would need to ship into the icon set's namespace |

Settled pattern (`features/browse/views/SourceIcon.tsx`):

- **Backdrop**: `surface-2` neutral, same across every source
- **Monogram**: 2 chars from `manifest.name` (camelCase-aware: HappyMH → HM)
- **Image override**: when manifest declares `icon` URL **and** the asset is ≥ 96px
- **Ring**: 1px `border-soft` inset for subtle depth

Aligns with Linear / Vercel / Notion / Plane icon systems.

## 10. Things we removed because they were wrong

Listed once so future agents don't reinvent them:

- `endpoints.feed: Record<string, BrowseEndpoint>` with arbitrary keys
  + `tabs` label override → became `endpoints.shelves: Shelf[]`
- `popular?: ListEndpoint` / `latest?: ListEndpoint` / `search?:
  ListEndpoint` distinct fields → unified into `shelves[]` + dedicated
  `search?: BrowseEndpoint`
- Per-host PROFILES dict in bunle-cdn → default Referer + per-endpoint
  override in manifest
- `X-Proxy-{Header}` allowlist of 8 names → base64 JSON blob via
  `X-Proxy-Headers` (URL form `?_h=`), denylist instead
- Sidebar "Cộng đồng" filter → Cộng đồng is a source in `/browse`
- Tab-based DA/non-DA host detection → single `VITE_PUBLIC_BASE_URL`
  env, hostname check kept only for routing decisions

When migrating manifests bump `useSources` persist key (`v3 → v4 →
v5 → …`) so stale localStorage doesn't preserve dead schema.

## 11. The reverse-engineering workflow

When adding a new source, follow [reverse-engineering-manga-sources.md](reverse-engineering-manga-sources.md).
The playbook documents the CDP eval helper, network capture script,
selector inspection ritual, and known-pitfalls list.

One pitfall worth repeating here: **gzip body bytes vs `len=0`**.
bunle-cdn passes upstream gzip through unchanged. `curl` without
`--compressed` shows 0 bytes for a 5KB JSON response. We spent an
hour debugging "Cloudflare gates this endpoint" before realising it
was just gzip. Always pass `Accept-Encoding: identity` (or
`--compressed`) when probing endpoints from the terminal.

## 12. When to delete legacy vs port it

From AGENTS.md: **delete dead legacy code before adding replacement
architecture that could be confused with it**.

Browse work has churned schema 7 times. Every change deletes:
- Old manifest fields
- Views that touched removed fields
- localStorage with stale shape (`persist` key bump)
- Wiki sections that document the dead shape

No `// deprecated, keep for compat` comments. No silent renames.
Schema migration is one PR; the codebase stays clean from one commit
to the next.
