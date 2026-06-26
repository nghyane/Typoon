# Manga source manifests

Each `*.json` here describes one upstream manga site. The browse-mode
runtime in the web app (`web/src/features/browse/manifest/`) reads these
declaratively — no executable code lives in a manifest.

## File layout

```
schema/source.schema.json   JSON Schema for one source manifest
index.json                  bundled-source list, shipped with the app
{id}.json                   one manifest per site
```

## Manifest shape

```jsonc
{
  "id":        "happymh",                        // unique, kebab-case
  "name":      "HappyMH",
  "host":      "m.happymh.com",                  // primary host (allowlist key)
  "language":  "zh",                             // BCP-47, used as source_lang default
  "version":   "0.1",
  "endpoints": {
    "popular": { ... },
    "search":  { ... },
    "manga":   { ... },
    "chapter": { ... }
  }
}
```

### Endpoint shapes

A **list endpoint** (`popular`, `search`) returns manga cards:

```jsonc
{
  "method":  "GET",
  "url":     "https://m.happymh.com/rank?d=day&page={page}",
  "parse":   "html",
  "list":    ".rank-list .item",          // CSS selector for each row
  "fields":  {
    "url":   "a@href",                    // selector@attribute
    "title": ".title",                    // text content
    "cover": "img@data-src"
  }
}
```

A **manga detail endpoint** returns metadata + chapter list:

```jsonc
{
  "method": "GET",
  "url":    "{mangaUrl}",
  "parse":  "html",
  "fields": {
    "title":       "h1.manga-title",
    "cover":       ".cover img@src",
    "description": ".manga-desc",
    "author":      ".manga-author",
    "status":      ".manga-status"
  },
  "chapters": {
    "list":   ".chapter-list li",
    "url":    "a@href",
    "number": "a@data-num",
    "title":  "a .name",
    "date":   ".date",
    "locked": "i.fa-lock@class"
  }
}
```

`locked` (optional) marks a chapter as **premium/locked**. Point it at a
marker that exists **only on locked rows** — typically a descendant lock or
coin icon, e.g. `i.fa-lock@class` or `.premium-block .coin@class`. Any
non-empty resolved value flags the chapter; locked chapters render greyed and
non-clickable in the chapter list and reader picker. Avoid a marker that is a
class on the row itself (use a descendant element instead), since the field
engine resolves selectors against descendants. Works on `chaptersApi` fields
and JSON sources too (e.g. `"locked": "is_premium"`).

A **chapter endpoint** returns the page URLs:

```jsonc
{
  "method": "GET",
  "url":    "{chapterUrl}",
  "parse":  "html",
  "list":   ".reader-img img",
  "fields": { "url": "@data-src" }
}
```

For JSON APIs (`parse: "json"`), the selector is a JSONPath
expression (`$.chapter.data[*]`) and attributes become dot-paths.

### Variables in URL templates

| Variable        | Source                                       |
|-----------------|----------------------------------------------|
| `{q}`           | user search query (URL-encoded)              |
| `{page}`        | 1-based page number                          |
| `{mangaUrl}`    | absolute manga URL (e.g. from a search hit)  |
| `{chapterUrl}`  | absolute chapter URL                         |
| `{mangaId}`     | extracted id (regex defined in manifest)     |
| `{chapterId}`   | extracted id                                 |

## Lọc theo thể loại (`filters`)

A source (or a single shelf) can expose genre/category filters. Filters live
either at the manifest top level (apply to every shelf + search) or on one
shelf via `shelf.filters` (apply only while that shelf is active):

```jsonc
"filters": [
  {
    "id":     "genre",
    "label":  "Thể loại",
    "type":   "select",        // "select" = one choice, "multi" = many
    "inject": "param",         // how the choice reaches the request — see below
    "options": [
      { "id": "action", "label": "Action", "param": "includedTags[]=…" },
      { "id": "18plus", "label": "18+",     "param": "rating=erotica", "nsfw": true }
    ]
  }
]
```

The selected options' `param` strings are spliced into the endpoint URL
according to `inject`:

| `inject`          | What happens                                                                 | URL placeholder |
|-------------------|------------------------------------------------------------------------------|-----------------|
| `param` (default) | `param`s joined with `&` (e.g. `&a=1&b=2`)                                    | `{filterParams}` |
| `path`            | the single selected `param` is dropped in as a **path segment** (select-only; always keeps one option active) | `{filterPath}` |
| `query`           | `param`s space-joined and folded into the search query `q` (e.g. nhentai `tag:"…"`) | rides existing `{q:q}` |

Notes:
- `path` filters need a `defaults` value (e.g. `"defaults": { "genre": "action" }`)
  so `{filterPath}` never resolves empty.
- An option with `"nsfw": true` renders as a standalone 18+ toggle chip, separate
  from the dropdown — use it for content-rating gates, not per-genre tagging.

## Adding a new source

1. Copy an existing manifest, change `id` / `host` / `language`.
2. Open one upstream page in DevTools, find stable CSS selectors.
3. Save as `{id}.json`, validate against `schema/source.schema.json`.
4. Add an entry to `index.json` so the app ships it as bundled.

## Hosting community sources

The web app can also install sources from any repo:

```
https://example.com/sources/
├── index.json     { "sources": [{ "id": "...", "url": "./xxx.json" }] }
├── xxx.json
└── icons/xxx.png
```

User pastes the `index.json` URL into Settings → Nguồn truyện → Thêm
nguồn. The app fetches, validates against the schema, and lets the
user install one or many sources.

Community sources run the same JSON runtime — no code, no eval. The
host must still pass the proxy allowlist (configured in
`bunle-cdn`).
