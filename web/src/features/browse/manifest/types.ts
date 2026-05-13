// Manifest types — schema v2, single source of truth across runtime,
// JSON validation, and the contributor README.
//
// One manifest = one upstream site. Manifests are *declarative* — no
// JavaScript runs from them, only selectors, URL templates, and a
// small set of typed normalisation primitives.
//
// Three sites drive the schema's shape:
//
//   * HappyMH  (zh)   — JSON for chapter list + reader, hardcoded
//                       `v=v4.x` magic param. Single language per site.
//   * OTruyen  (vi)   — REST JSON for everything; manga detail
//                       inlines the chapter list; chapter pages come
//                       from a *second* API server (`domain_cdn`).
//                       Cover URL = `{cdn}/uploads/comics/{thumb}`.
//   * MangaDex (multi)— REST JSON; cover comes from a relationship
//                       blob in the same response; multi-language
//                       chapter feed (user picks lang).
//
// The schema's job is to express all three with the same vocabulary.

// ─── core building blocks ──────────────────────────────────────────

export type ParseMode = 'html' | 'json'

/** Selector grammar (see selectors.ts):
 *
 *    css.selector            HTML text
 *    css.selector@attr       HTML attribute
 *    @attr                   attribute of the current row root
 *    $.json.path[*]@key      JSONPath + optional key
 *    =template-with-{vars}   composite from other resolved fields
 */
export type Selector = string

export interface HttpRequest {
  method?:  'GET' | 'POST'
  /** URL template with `{var}` placeholders. Vars come from:
   *    1. caller args (mangaUrl, chapterUrl, q, page, …)
   *    2. `extract` regex named groups
   *    3. resolved field values inside the SAME endpoint (templates only) */
  url:      string
  headers?: Record<string, string>
  body?:    string
  parse:    ParseMode                   // explicit, no inferring
}

// ─── pagination ────────────────────────────────────────────────────

/** How the source paginates a list endpoint. The runtime increments
 *  `{page}` or `{offset}` between fetches; if a page returns fewer
 *  rows than `pageSize`, paging stops. */
export interface Pagination {
  /** `page`   — `{page}` starts at 1 and increments by 1.
   *  `offset` — `{offset}` starts at 0 and increments by `pageSize`. */
  type:     'page' | 'offset'
  pageSize: number
}

// ─── filters ───────────────────────────────────────────────────────

/** User-facing filter. The runtime turns the selected option(s) into
 *  a `{filterParams}` string (concatenated `&key=value` fragments)
 *  the endpoint URL template references.
 *
 *  Defaults come from `manifest.defaults`. URL state mirrors the
 *  selected option ids so views are shareable. */
export interface Filter {
  id:    string
  label: string
  /** `select` — one option at a time (sort, status, …)
   *  `multi`  — many options at once (languages, tags, …) */
  type:  'select' | 'multi'
  options: FilterOption[]
}

export interface FilterOption {
  id:    string
  label: string
  /** Raw query fragment(s) appended to the request URL when this
   *  option is active. Multiple `key=value` separated by `&`, no
   *  leading `&`. Empty string = no-op (e.g. "any" option). */
  param: string
}

// ─── endpoints ─────────────────────────────────────────────────────

/** Browse / search endpoint. Returns a paginated grid of manga. */
export interface BrowseEndpoint extends HttpRequest {
  pagination?: Pagination
  /** JSONPath / CSS selector returning N rows. */
  list:        Selector
  /** Each row produces a MangaSummary via this field map. */
  fields:      {
    /** Stable id (also serves as `mangaUrl` for downstream calls). */
    url:    Selector
    title:  Selector
    cover?: Selector
  }
  /** Extra fields the manifest may resolve to feed `cover` template
   *  (e.g. MangaDex `fileName` from relationships).  Resolved
   *  BEFORE `fields.cover` so cover templates can reference them. */
  extras?: Record<string, Selector>
}

/** Manga detail endpoint. Returns metadata + (optionally) inline
 *  chapter list. If chapters are inline, omit `chaptersApi`. */
export interface MangaEndpoint extends HttpRequest {
  /** Regex with named groups, matched against `mangaUrl`. The
   *  captured groups become template vars for THIS endpoint and
   *  every endpoint chained from it (chaptersApi). */
  extract?: string
  fields: {
    title:        Selector
    cover?:       Selector
    description?: Selector
    author?:      Selector
    status?:      Selector
  }
  extras?: Record<string, Selector>
  /** Inline chapter list when the manga response carries it. */
  chapters?: ChapterListSpec
}

/** Standalone chapter-list endpoint. Used when the manga page does
 *  not inline the list (HappyMH, MangaDex). Receives the same vars
 *  as the manga endpoint (`mangaUrl` + extracted groups). */
export interface ChaptersApiEndpoint extends HttpRequest {
  pagination?: Pagination
  list:        Selector
  fields:      ChapterFields
  /** Per-endpoint override of `manifest.chapterNumberNorm`. Use when
   *  this endpoint emits chapters in a different shape than the rest
   *  of the source. */
  chapterNumberNorm?: ChapterNumberNorm
  /** Optional predicate fields evaluated per-row. The row is kept
   *  only when every predicate returns a truthy, non-empty value.
   *  Use for filtering out external chapters, locked previews, etc.
   *
   *  Example (MangaDex skip external):
   *    `keepIf: { hasPages: "@attributes.pages" }`
   *  → drops rows where `attributes.pages` is 0 / null / "". */
  keepIf?: Record<string, Selector>
}

export interface ChapterListSpec {
  list:   Selector
  fields: ChapterFields
  /** Per-endpoint override of `manifest.chapterNumberNorm`. */
  chapterNumberNorm?: ChapterNumberNorm
}

export interface ChapterFields {
  /** Stable chapter id and URL the reader navigates to. The reader
   *  passes this back into the chapter endpoint via `chapterUrl`. */
  url:      Selector
  /** Primary numeric/short identifier (e.g. "12", "12.5", "Vol.3 Ch.4").
   *  Used as default UI label prefix and for sort ordering.
   *  Sources where the chapter has no separate number (HappyMH:
   *  the only label is "第106话") should pass it here verbatim. */
  number?:  Selector
  /** Optional secondary label ("Hồi kết", "Dungeon Boss"). */
  title?:   Selector
  date?:    Selector
  /** BCP-47 — when the source has multi-language chapters and the
   *  user filters by language in the UI. */
  language?: Selector
  /** Explicit display template. When present, the UI renders this
   *  verbatim (no "Chương " prefix). Use when the source's number
   *  field already contains a fully-formed label (e.g. HappyMH
   *  `chapterName: "第106话"`). Supports `{number}`, `{title}`,
   *  `{date}`, `{language}` interpolation if combining with other
   *  fields. */
  label?:   Selector
}

/** Declarative chapter-number normalisation primitives.
 *
 *  Maps the raw `number` (and/or `label`) a source publishes to the
 *  canonical `work_chapter.number_norm` used to dedupe chapters
 *  across sources of the same Work. No JavaScript runs — every
 *  primitive is a typed transform the runtime knows how to evaluate.
 *
 *  Evaluation order (per chapter row):
 *    1. Pick the input string from `input` (default: `number`).
 *    2. Walk `patterns` in priority order. First regex that matches
 *       wins; capture group 1 (if present) is used, else the whole
 *       match.
 *    3. If no pattern matched, fall back to `default`.
 *    4. Apply every step in `postprocess` to the resulting string.
 *
 *  Sources that need only the global default (extract first
 *  number-like substring, strip leading zeros, lowercase, slug
 *  fallback) may omit this field entirely. */
export interface ChapterNumberNorm {
  /** Which raw field to read. `number` (default) is almost always
   *  right; pick `label` only when the source publishes the canonical
   *  number inside the long label (e.g. HappyMH's `chapterName`). */
  input?:       'number' | 'label'
  /** Regex strings tried top-down. Compile-time-validated by the
   *  loader; invalid regex rejects the whole manifest. Capture group
   *  1, when present, is the extracted number. */
  patterns?:    string[]
  /** What to emit when no pattern matched. Defaults to `'slug'`. */
  default?:     'slug' | 'empty' | 'verbatim'
  /** Post-processing applied in order to the extracted string. */
  postprocess?: ('lowercase' | 'trim' | 'stripLeadingZeros')[]
}

/** Chapter-pages endpoint. Returns the upstream URLs of every page.
 *
 *  Two output shapes:
 *
 *    1. `list` + `fields.url` — each row already has a URL.
 *    2. `pages.iterate` + `pages.template` — root response has a
 *       baseUrl + hash + array of file names; runtime maps each
 *       file through the template.
 *
 *  Only one of `list` / `pages` may be set. */
export interface ChapterEndpoint extends HttpRequest {
  extract?: string
  list?:    Selector
  fields?:  { url: Selector }
  pages?:   {
    /** Field map resolved against the response root; the final
     *  field "files" must produce an array (JSONPath wildcard). */
    extras:   Record<string, Selector>
    /** Name of the array field inside `extras`. */
    iterate:  string
    /** Per-iteration template; `{file}` plus any other extras. */
    template: string
  }
}

// ─── source manifest ───────────────────────────────────────────────

export interface SourceManifest {
  id:        string
  name:      string
  /** Primary host — also the proxy allowlist key.
   *  For `kind: 'internal'` sources, this is informational only. */
  host:      string
  /** Human site URL (footer "open original" links). */
  homepage?: string
  /** BCP-47 codes the site provides chapters in. `["multi"]` for a
   *  multi-language source where the language filter is required. */
  languages: string[]
  /** Source plumbing:
   *
   *    `external`  (default) — JSON/HTML manifest, fetched via the
   *                bunle-cdn proxy. Endpoints describe upstream
   *                manga sites (HappyMH, MangaDex, OTruyen).
   *
   *    `internal`  — wired to typoon's own backend. Manga URLs are
   *                project routes (`/projects/{id}`); chapters open
   *                in the regular reader. `endpoints` is ignored —
   *                runtime branches on `kind` and calls `api.*`. */
  kind?:     'external' | 'internal'
  nsfw?:     boolean
  version:   string

  /** Visual identity for source tiles. Two options:
   *
   *    1. Image override — declare `icon` (relative or absolute URL).
   *       Manifest authors with a real brand asset should use this.
   *       Best at 256×256, transparent background; rendered square.
   *
   *    2. Monogram (default) — runtime composes a 2-char tile from
   *       the source name. Tint the background via `accent` (Tailwind
   *       semantic name OR raw hex). When omitted, derived from
   *       `id` hash so the same source always gets the same colour.
   *
   *  Use `accent: 'auto'` to opt out of override and force the hash
   *  fallback explicitly (e.g. for community source variants). */
  icon?:   string
  accent?: string

  /** Default filter selections per filter id. Each value is either
   *  one option id (for `select`) or an array of ids (for `multi`). */
  defaults?: Record<string, string | string[]>

  /** Filters/sorts shown in the source feed bar. */
  filters?: Filter[]

  /** Source-wide chapter number normalisation. The runtime applies
   *  this spec to every `ChapterFields.number` it extracts (unless
   *  the endpoint overrides via `ChaptersApiEndpoint.chapterNumberNorm`
   *  or `ChapterListSpec.chapterNumberNorm`). Omit to inherit the
   *  global default — see runtime.ts `DEFAULT_CHAPTER_NUMBER_NORM`. */
  chapterNumberNorm?: ChapterNumberNorm

  /** Required for `kind: 'external'`. Internal sources branch in
   *  runtime and ignore this field — leave a stub `{ shelves: [] }`
   *  in the JSON to satisfy schema validation, or omit if the
   *  loader accepts undefined. */
  endpoints?: {
    /** Shelves shown on the source landing — each renders a horizontal
     *  rail of cards. Order matters: top shelf = highest priority for
     *  the user's intent. A "Xem tất cả" link on each shelf opens a
     *  full grid view at /browse/$source/shelf/$shelfId. */
    shelves: Shelf[]
    /** Search endpoint — activates when the user types a query. */
    search?: BrowseEndpoint
    manga:   MangaEndpoint
    /** Required when `manga.chapters` is absent. */
    chaptersApi?: ChaptersApiEndpoint
    chapter:  ChapterEndpoint
  }
}

/** One shelf in the source landing — a labelled horizontal rail.
 *
 *  Examples:
 *    { id: "popular-day", label: "Phổ biến hôm nay", endpoint: {...} }
 *    { id: "latest",      label: "Mới cập nhật",     endpoint: {...} }
 *
 *  `id` is the URL segment for the full-grid detail page
 *  (`/browse/$source/shelf/$id`). Keep stable; renames invalidate
 *  bookmarks and Continue-Reading mapping. */
export interface Shelf {
  id:        string
  label:     string
  /** Optional short caption rendered under the label. */
  hint?:     string
  endpoint:  BrowseEndpoint
}

// ─── runtime result types (unchanged contract for callers) ─────────

export interface MangaSummary {
  id:    string
  url:   string
  title: string
  cover: string | null
}

export interface MangaChapterRef {
  id:        string
  url:       string
  /** Pre-composed display string from the manifest. UI renders this
   *  verbatim instead of stitching number+title together (avoids
   *  duplication when a source's "number" already contains the
   *  full label). */
  label:     string
  number:    string
  /** Canonical key used to dedupe this chapter against sibling
   *  sources of the same Work. Computed at fetch time by the
   *  manifest runtime — applies the source's `chapterNumberNorm`
   *  spec (or the global default when absent). Server treats this
   *  value as opaque; it must be deterministic per (manifest,
   *  raw number) so the same chapter always maps to the same key. */
  numberNorm: string
  title:     string | null
  date:      string | null
  language:  string | null
}

export interface MangaDetail extends MangaSummary {
  description: string | null
  author:      string | null
  status:      string | null
  /** Languages with at least one chapter on the source. UI uses this
   *  to narrow the `manifest.languages` picker so the user doesn't
   *  pick a lang that returns 0 chapters. Falls back to
   *  `manifest.languages` when null. */
  availableLanguages: string[] | null
  chapters:    MangaChapterRef[]
}

export interface ChapterPages {
  url:   string
  pages: string[]
}

// ─── installed source registry ────────────────────────────────────

export type SourceOrigin = 'bundled' | 'repo' | 'file'

export interface InstalledSource {
  manifest:    SourceManifest
  origin:      SourceOrigin
  repoUrl?:    string
  author?:     string
  installedAt: number
  enabled:     boolean
}
