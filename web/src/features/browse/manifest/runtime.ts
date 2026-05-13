// Manifest runtime — execute endpoints declared in a SourceManifest
// (schema v2). Single responsibility per public function:
//
//   fetchBrowse        — paginated grid of MangaSummary (feed or search)
//   fetchMangaDetail   — single MangaDetail with chapter list
//   fetchChapterPages  — list of upstream image URLs for a chapter
//
// All network goes through `pfetch` (proxy + Referer/UA per host).

import { pfetch } from '../proxy'
import {
  queryHtmlOne, queryHtmlAll, queryJsonOne, queryJsonAll,
} from './selectors'
import {
  applyChapterNumberNorm, compileChapterNumberNorm,
} from './normalize'
import type {
  BrowseEndpoint, ChapterListSpec, ChaptersApiEndpoint,
  ChapterFields, ChapterPages, HttpRequest, MangaChapterRef, MangaDetail,
  MangaSummary, SourceManifest, Shelf as ShelfManifestEntry,
} from './types'

// The 'internal' source concept (Community) was removed in the v5
// architecture — Hội Mê Truyện is a guild-scoped feed, not a manifest
// source. This kept-around helper lets call sites stay short-circuit
// safe in case a stale persisted source slips through.
function isInternal(manifest: SourceManifest): boolean {
  return manifest.kind === 'internal'
}

// ─── template + URL helpers ───────────────────────────────────────

export type Vars = Record<string, string | number | undefined | null>

/** Substitute `{name}` tokens.
 *
 *  Rules:
 *    - `{name}`     verbatim. If the value looks like an absolute URL
 *                   (`http://`, `https://`) it stays as-is — common
 *                   when feeding `{mangaUrl}` as the entire request
 *                   URL. Otherwise interpolated raw.
 *    - `{name:q}`   URL-encoded (used for query-string fragments
 *                   where the value may contain `&`, `=`, spaces, …).
 *
 *  Missing variable → empty string. */
function tpl(template: string, vars: Vars): string {
  return template.replace(/\{([a-zA-Z_][\w]*)(?::(q))?\}/g, (_, name, mod) => {
    const v = vars[name]
    if (v == null) return ''
    const s = String(v)
    if (mod === 'q') return encodeURIComponent(s)
    return s
  })
}

function absolutise(href: string | null, base: string): string | null {
  if (!href) return null
  try { return new URL(href, base).href } catch { return null }
}

function extractVars(input: string, pattern?: string): Record<string, string> {
  if (!pattern) return {}
  try { return (new RegExp(pattern).exec(input)?.groups ?? {}) as Record<string, string> }
  catch { return {} }
}

// ─── HTTP layer ───────────────────────────────────────────────────

interface FetchResult {
  url:    string
  parsed: Document | unknown
}

async function exec(req: HttpRequest, vars: Vars): Promise<FetchResult> {
  const url = tpl(req.url, vars)
  const res = await pfetch(url, {
    headers: req.headers,
    init: {
      method: req.method ?? 'GET',
      body:   req.body && tpl(req.body, vars),
    },
  })
  if (!res.ok) throw new Error(`HTTP ${res.status} on ${url}`)
  if (req.parse === 'json') {
    return { url, parsed: await res.json() }
  }
  const text = await res.text()
  return { url, parsed: new DOMParser().parseFromString(text, 'text/html') }
}

// ─── field rows (HTML element or JSON node) ───────────────────────

interface Row {
  get: (selector: string) => string | null
  /** For JSON rows: the raw node, used when a list endpoint needs to
   *  pass the row into a sub-template (e.g. iterate). */
  raw: unknown
}

function htmlRow(el: Element): Row {
  return { get: (s) => queryHtmlOne(el, s), raw: el }
}

function jsonRow(node: unknown): Row {
  return {
    get: (s) => {
      const v = queryJsonOne(node, s)
      if (v == null) return null
      // Preserve array/object shape for fields that need it
      // (availableTranslatedLanguages etc). `String([...])` would
      // produce `ka,ru,…` which loses type info; JSON.stringify keeps
      // a parseable representation the caller can recover.
      if (typeof v === 'object') return JSON.stringify(v)
      return String(v)
    },
    raw: node,
  }
}

function rootRow(parsed: Document | unknown, mode: 'html' | 'json'): Row {
  return mode === 'json'
    ? jsonRow(parsed)
    : { get: (s) => queryHtmlOne(parsed as Document, s), raw: parsed }
}

function rowsFrom(parsed: unknown, list: string, mode: 'html' | 'json'): Row[] {
  return mode === 'json'
    ? queryJsonAll(parsed, list).map(jsonRow)
    : queryHtmlAll(parsed as Document, list).map(htmlRow)
}

// ─── field resolution ─────────────────────────────────────────────

/** Two-pass resolve so template order in JSON doesn't matter:
 *    1. selectors (everything that doesn't start with `=`)
 *    2. templates (`=...`) — can reference selector results + globals */
function resolveFields(
  row: Row,
  fields: Record<string, string>,
  globals: Vars,
): Record<string, string | null> {
  const out: Record<string, string | null> = {}
  for (const [k, decl] of Object.entries(fields)) {
    if (!decl.startsWith('=')) out[k] = row.get(decl)
  }
  for (const [k, decl] of Object.entries(fields)) {
    if (decl.startsWith('=')) {
      out[k] = tpl(decl.slice(1), { ...globals, ...out })
    }
  }
  return out
}

// ─── public API: browse ───────────────────────────────────────────

export interface BrowseArgs {
  /** Search query (used when calling `endpoints.search`). */
  q?:     string
  /** 1-based page; runtime translates to offset if endpoint paginates by offset. */
  page?:  number
  /** Resolved filter param fragments — already concatenated with `&`. */
  filterParams?: string
}

// ─── source-facing helpers (views consume these instead of touching
//     `manifest.endpoints` directly) ─────────────────────────────────

/** A normalised shelf descriptor used by views. Both external
 *  manifests and internal sources surface their shelves through this
 *  shape so render code doesn't branch on `kind`. */
export interface ShelfDescriptor {
  id:    string
  label: string
  hint?: string
  /** Whether the shelf supports a "Xem tất cả" deep view. Internal
   *  shelves always do; external shelves do when their endpoint
   *  paginates. */
  paginated: boolean
}

/** Enumerate shelves regardless of source kind. */
export function getShelves(manifest: SourceManifest): ShelfDescriptor[] {
  if (isInternal(manifest)) return []
  return (manifest.endpoints?.shelves ?? []).map((s: ShelfManifestEntry) => ({
    id:        s.id,
    label:     s.label,
    hint:      s.hint,
    paginated: !!s.endpoint.pagination,
  }))
}

/** Whether a manifest exposes search at all. */
export function hasSearch(manifest: SourceManifest): boolean {
  if (isInternal(manifest)) return false
  return !!manifest.endpoints?.search
}

/** Page size for a shelf, when known. Used by infinite-scroll
 *  termination. Returns `Infinity` for shelves without pagination
 *  (HappyMH /rank/day returns the whole list in one shot). */
export function shelfPageSize(
  manifest: SourceManifest, shelfId: string,
): number {
  if (isInternal(manifest)) return 24
  const shelf = manifest.endpoints?.shelves.find((s) => s.id === shelfId)
  return shelf?.endpoint.pagination?.pageSize ?? Infinity
}

export function searchPageSize(manifest: SourceManifest): number {
  return manifest.endpoints?.search?.pagination?.pageSize ?? Infinity
}

// ═══════════════════════════════════════════════════════════════════
// Fetch dispatch
// ═══════════════════════════════════════════════════════════════════

/** Single dispatch entry for shelf / search browse.
 *
 *  Caller passes the manifest + shelf id (or `{ search: true }`) and
 *  this function picks the right backing fetcher:
 *
 *    • `kind: 'internal'` → routes to typoon's own /api/projects via
 *      `internal.fetchInternalBrowse`.
 *    • `kind: 'external'` (default) → executes the BrowseEndpoint
 *      against the source manifest using the proxy.
 *
 *  View code never branches on `kind`; everything else stays the
 *  same shape (`MangaSummary[]`). */
export async function fetchBrowse(
  manifest: SourceManifest,
  shelfId: string | { search: true },
  args: BrowseArgs = {},
): Promise<MangaSummary[]> {
  if (isInternal(manifest)) return []
  const endpoint =
    typeof shelfId === 'string'
      ? manifest.endpoints?.shelves.find((s) => s.id === shelfId)?.endpoint
      : manifest.endpoints?.search
  if (!endpoint) return []
  return execBrowseEndpoint(endpoint, args)
}

async function execBrowseEndpoint(
  endpoint: BrowseEndpoint,
  args: BrowseArgs,
): Promise<MangaSummary[]> {
  const page = args.page ?? 1
  const offset = endpoint.pagination?.type === 'offset'
    ? (page - 1) * endpoint.pagination.pageSize
    : 0
  const vars: Vars = {
    q:           args.q ?? '',
    page,
    offset,
    filterParams: args.filterParams ?? '',
  }
  const { url, parsed } = await exec(endpoint, vars)
  const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)

  const out: MangaSummary[] = []
  for (const row of rows) {
    const extras = endpoint.extras
      ? resolveFields(row, endpoint.extras, vars)
      : {}
    const f = resolveFields(row, endpoint.fields as Record<string, string>,
                            { ...vars, ...extras })
    const upstreamUrl = absolutise(f.url, url)
    if (!upstreamUrl) continue
    out.push({
      id:    upstreamUrl,
      url:   upstreamUrl,
      title: f.title ?? '(không tên)',
      cover: absolutise(f.cover ?? null, url),
    })
  }
  return out
}

// ─── public API: manga detail ─────────────────────────────────────

export async function fetchMangaDetail(
  manifest: SourceManifest,
  mangaUrl: string,
  args: { language?: string } = {},
): Promise<MangaDetail> {
  // Internal sources don't go through manga-detail at all — the
  // routing layer redirects to /projects/{id}. Guard so the helper
  // never produces a half-empty MangaDetail.
  if (isInternal(manifest) || !manifest.endpoints?.manga) {
    throw new Error(`Source ${manifest.id} has no manga endpoint`)
  }
  const endpoint = manifest.endpoints.manga
  const vars: Vars = {
    mangaUrl,
    language: args.language ?? manifest.languages[0],
    ...extractVars(mangaUrl, endpoint.extract),
  }

  const { url: baseUrl, parsed } = await exec(endpoint, vars)
  const root = rootRow(parsed, endpoint.parse)
  const extras = endpoint.extras
    ? resolveFields(root, endpoint.extras, vars)
    : {}
  const f = resolveFields(root, endpoint.fields as Record<string, string>,
                          { ...vars, ...extras })

  // Chapters: inline (manga.chapters) or external (chaptersApi).
  let chapters: MangaChapterRef[] = []
  if (endpoint.chapters) {
    chapters = collectChapters(parsed, endpoint.parse, endpoint.chapters,
                               baseUrl, { ...vars, ...extras }, manifest)
  } else if (manifest.endpoints?.chaptersApi) {
    chapters = await fetchChaptersExternal(
      manifest.endpoints.chaptersApi,
      { ...vars, ...extras },
      manifest,
    )
  }

  return {
    id:                 mangaUrl,
    url:                mangaUrl,
    title:              f.title ?? '(không tên)',
    cover:              absolutise(f.cover ?? null, baseUrl),
    description:        f.description ?? null,
    author:             f.author      ?? null,
    status:             f.status      ?? null,
    availableLanguages: parseLangList(f.availableLangs),
    chapters,
  }
}

/** Parse a stringified array value back into `string[]`. Selector
 *  engine flattens everything to string; for array-shaped fields the
 *  manifest names them with the `*Langs` / `*List` convention and
 *  runtime restores. Falls back to `null` so callers can use the
 *  manifest's static `languages` list. */
function parseLangList(raw: string | null | undefined): string[] | null {
  if (!raw) return null
  // JSON-stringified array (`["ka","ru"]`) — common when the field
  // selector resolves to a JSON array node.
  if (raw.startsWith('[')) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        return parsed.filter((x): x is string => typeof x === 'string')
      }
    } catch { /* fall through */ }
  }
  // Comma- or space-separated fallback.
  const parts = raw.split(/[\s,]+/).filter((s) => s.length > 0)
  return parts.length > 0 ? parts : null
}

function collectChapters(
  parsed: unknown,
  parseMode: 'html' | 'json',
  spec: ChapterListSpec,
  baseUrl: string,
  globals: Vars,
  manifest: SourceManifest,
): MangaChapterRef[] {
  const rows = rowsFrom(parsed, spec.list, parseMode)
  const norm = compileChapterNumberNorm(
    spec.chapterNumberNorm ?? manifest.chapterNumberNorm,
  )
  return rows
    .map((r) => buildChapter(r, spec.fields, baseUrl, globals, norm))
    .filter((c): c is MangaChapterRef => c !== null)
}

async function fetchChaptersExternal(
  endpoint: ChaptersApiEndpoint,
  vars: Vars,
  manifest: SourceManifest,
): Promise<MangaChapterRef[]> {
  // Single-page fetch for now. Pagination support is trivial to add
  // when a source needs it: loop until rows < pageSize.
  const { url, parsed } = await exec(endpoint, { ...vars, page: 1, offset: 0 })
  const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)
  const kept = endpoint.keepIf
    ? rows.filter((r) => passesPredicates(r, endpoint.keepIf!, vars))
    : rows
  const norm = compileChapterNumberNorm(
    endpoint.chapterNumberNorm ?? manifest.chapterNumberNorm,
  )
  return kept
    .map((r) => buildChapter(r, endpoint.fields, url, vars, norm))
    .filter((c): c is MangaChapterRef => c !== null)
}

/** Every predicate must resolve to a non-empty value (per `isEmpty`
 *  rules: null, "", "0", []). Used for row filtering. */
function passesPredicates(
  row: Row, preds: Record<string, string>, globals: Vars,
): boolean {
  for (const decl of Object.values(preds)) {
    const v = decl.startsWith('=')
      ? tpl(decl.slice(1), { ...globals })
      : row.get(decl)
    if (v == null) return false
    const s = String(v).trim()
    if (s.length === 0 || s === '0' || s === 'false' || s === 'null') return false
  }
  return true
}

function buildChapter(
  row: Row, fields: ChapterFields, baseUrl: string, globals: Vars,
  norm: ReturnType<typeof compileChapterNumberNorm>,
): MangaChapterRef | null {
  const f = resolveFields(row, fields as unknown as Record<string, string>, globals)
  const url = absolutise(f.url, baseUrl)
  if (!url) return null
  const number = f.number ?? ''
  const title  = f.title  ?? null
  const label  = f.label  ?? composeChapterLabel(number, title)
  const numberNorm = applyChapterNumberNorm(norm, { number, label })
  return {
    id:         url,
    url,
    number,
    numberNorm,
    title,
    label,
    date:       f.date      ?? null,
    language:   f.language  ?? null,
    scanlator:  f.scanlator ?? null,
  }
}

/** Default chapter label when the manifest does not specify one.
 *  - "Chương N · title"   when both number and title are present.
 *  - "Chương N"           when only the number.
 *  - "title"              when only the title.
 *  - "?"                  when neither.
 *
 *  Sources where `number` already contains a fully-formed label
 *  (HappyMH `chapterName: "第106话"`) should declare `label: "@..."` */
function composeChapterLabel(number: string, title: string | null): string {
  const n = number.trim()
  const t = title?.trim() ?? ''
  if (n && t) return `Chương ${n} · ${t}`
  if (n)      return `Chương ${n}`
  if (t)      return t
  return '?'
}

// ─── public API: chapter pages ────────────────────────────────────

export async function fetchChapterPages(
  manifest: SourceManifest,
  chapterUrl: string,
): Promise<ChapterPages> {
  if (isInternal(manifest) || !manifest.endpoints?.chapter) {
    throw new Error(`Source ${manifest.id} has no chapter endpoint`)
  }
  const endpoint = manifest.endpoints.chapter
  const vars: Vars = {
    chapterUrl,
    ...extractVars(chapterUrl, endpoint.extract),
  }
  const { url: requestUrl, parsed } = await exec(endpoint, vars)

  // Shape 1: list of rows each carrying a URL.
  if (endpoint.list && endpoint.fields) {
    const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)
    const pages: string[] = []
    for (const r of rows) {
      const f = resolveFields(r,
        endpoint.fields as unknown as Record<string, string>,
        vars,
      )
      const u = absolutise(f.url, requestUrl)
      if (u) pages.push(u)
    }
    return { url: chapterUrl, pages }
  }

  // Shape 2: root has baseUrl/hash + array of files → template each.
  if (endpoint.pages) {
    const root = rootRow(parsed, endpoint.parse)
    const spec = endpoint.pages
    // Resolve every extra EXCEPT the iterate-target as a scalar — the
    // iterate-target is the list selector and stays an array.
    const scalarExtras: Record<string, string | null> = {}
    for (const [k, sel] of Object.entries(spec.extras)) {
      if (k === spec.iterate) continue
      scalarExtras[k] = root.get(sel)
    }
    const filesSel = spec.extras[spec.iterate]
    if (!filesSel) {
      throw new Error(`chapter.pages.iterate "${spec.iterate}" not in extras`)
    }
    const files = endpoint.parse === 'json'
      ? queryJsonAll(parsed, filesSel)
      : queryHtmlAll(parsed as Document, filesSel).map((el) => el.textContent)
    const pages: string[] = []
    for (const file of files) {
      const f = typeof file === 'string' ? file : String(file ?? '')
      if (!f) continue
      pages.push(tpl(spec.template, { ...vars, ...scalarExtras, file: f }))
    }
    return { url: chapterUrl, pages }
  }

  throw new Error(`chapter endpoint missing list/pages spec (${manifest.id})`)
}

// ─── filter param assembly (used by views) ────────────────────────

/** Build a `&key=value&...` fragment from the selected filter state.
 *  Empty when nothing is selected — caller passes it as
 *  `{filterParams}` into the endpoint URL template. */
export function assembleFilterParams(
  manifest: SourceManifest,
  state: Record<string, string | string[]>,
): string {
  if (!manifest.filters) return ''
  const fragments: string[] = []
  for (const filter of manifest.filters) {
    const sel = state[filter.id]
    if (sel == null) continue
    const ids = Array.isArray(sel) ? sel : [sel]
    for (const id of ids) {
      const opt = filter.options.find((o) => o.id === id)
      if (opt?.param) fragments.push(opt.param)
    }
  }
  return fragments.length > 0 ? '&' + fragments.join('&') : ''
}
