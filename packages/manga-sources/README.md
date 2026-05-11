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
    "date":   ".date"
  }
}
```

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
