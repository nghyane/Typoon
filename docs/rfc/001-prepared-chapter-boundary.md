# RFC-001: PreparedChapter Boundary

## Status

Accepted

## Problem

v2 lets raw source pages enter the main translation pipeline. To handle webtoon
continuity it stitches a full chapter strip, scans it, then reconstructs logical
pages later through `LazyPageProvider`.

This caused the main failure modes:

- duplicate normalization between scan and render
- fragile page reconstruction
- coordinate ambiguity
- hidden memory cost
- cross-page logic leaking into scan/render
- surface fixes instead of root-cause fixes

## Decision

The main pipeline accepts only a `PreparedChapter`.

```text
RawSource -> prepare -> PreparedChapter -> page-local pipeline -> RenderedChapter
```

`prepare` handles raw source quirks. The page-local pipeline never stitches,
recuts, or reconstructs pages from raw source.

## PreparedChapter format

```text
PreparedChapter/
  manifest.json
  pages/
    0000.png
    0001.png
    0002.png
```

Example manifest:

```json
{
  "version": 1,
  "source": "raw folder or url",
  "page_count": 3,
  "pages": [
    { "index": 0, "file": "pages/0000.png", "width": 830, "height": 3500 }
  ]
}
```

## Prepare responsibilities

- sort raw files
- group consecutive compatible widths
- stitch each group temporarily
- find safe cuts by row cost
- export prepared pages
- write manifest
- write visual debug artifacts

Prepare is the only stage allowed to handle cross-file continuity.

## Main pipeline responsibilities

For each prepared page:

```text
detect -> group -> OCR -> classify -> translate -> layout -> render
```

No main-pipeline stage may read raw source files after preparation.

## Non-goals

- No Rust rewrite in this RFC.
- No storage/server redesign.
- No cross-page bubble merging inside scan/render.

