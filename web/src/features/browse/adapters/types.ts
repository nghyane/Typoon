// Source adapter interface.
//
// Adapters handle sites whose chapter-page loading requires imperative
// logic that cannot be expressed in the declarative manifest schema
// (per-page extension maps, JS-evaluated image URLs, etc.).
//
// Optional overrides (omit to fall back to declarative runtime):
//   fetchChapterPages  — required when declared; returns URLs or tokens
//   resolvePageUrl     — lazy per-page URL resolution (see below)
//   fetchMangaDetail   — override gallery metadata fetching
//   fetchBrowse        — override listing for JS-rendered / binary-index sites
//
// Adapters run in the browser (same context as the manifest runtime)
// and share the same proxy (`pfetch`) for network access.

import type { BrowseArgs, ChapterPages, MangaDetail, MangaSummary, SourceManifest } from '../manifest/types'

export interface SourceAdapter {
  fetchChapterPages(
    manifest:    SourceManifest,
    chapterUrl:  string,
    userCookies: Record<string, string>,
  ): Promise<ChapterPages>

  /** Lazily resolve one page URL from an opaque token returned by
   *  `fetchChapterPages` (via `ChapterPages.tokens`). The reader
   *  calls this per-page when the slot enters the viewport, keyed
   *  by token in React Query, so each URL is fetched exactly once
   *  and cached independently.
   *
   *  Use when the source requires one network call per image (signed
   *  CDN tokens, showpage-style APIs). Without this, `fetchChapterPages`
   *  must resolve all N URLs upfront, blocking the reader until done. */
  resolvePageUrl?(
    manifest:    SourceManifest,
    token:       string,
    userCookies: Record<string, string>,
  ): Promise<string>

  fetchMangaDetail?(
    manifest:    SourceManifest,
    mangaUrl:    string,
    userCookies: Record<string, string>,
  ): Promise<MangaDetail>

  /** Override the entire browse/search flow for sites whose listing
   *  cannot be expressed declaratively (fully JS-rendered, binary
   *  index protocols, etc.). When present, the declarative manifest
   *  `endpoints.shelves` / `endpoints.search` are ignored. */
  fetchBrowse?(
    manifest:    SourceManifest,
    shelfId:     string | { search: true },
    args:        BrowseArgs,
  ): Promise<MangaSummary[]>
}
