// Manifest runtime — execute endpoints declared in a SourceManifest
// (schema v2/v3). Single responsibility per public function:
//
//   fetchBrowse        — paginated grid of MangaSummary (feed or search)
//   fetchMangaDetail   — single MangaDetail with chapter list
//   fetchChapterPages  — list of upstream image URLs for a chapter
//
// All network goes through `pfetch` (proxy + Referer/UA per host).
//
// v3 additions (backward compatible):
//   • rootExtras  — resolved once on response root, shared across rows
//   • keepIf      — row filter on browse endpoints (was only chaptersApi)
//   • pages.count — numeric range 1..N for chapter pages
//   • authRequired / cookieNames — cookie injection via proxy

import { fetchSource } from '../proxy'
import {
  queryHtmlOne, queryHtmlAll, queryJsonOne, queryJsonAll,
} from './selectors'
import {
  applyChapterNumberNorm, compileChapterNumberNorm,
} from './normalize'
import { getAdapter } from '../adapters'
import type {
  BrowseEndpoint, BrowseArgs, ChapterListSpec, ChaptersApiEndpoint,
  ChapterFields, ChapterPages, HttpRequest, MangaChapterRef, MangaDetail,
  MangaSummary, SourceManifest, Shelf as ShelfManifestEntry,
} from './types'

function isInternal(manifest: SourceManifest): boolean {
  return manifest.kind === 'internal'
}

// ─── template + URL helpers ───────────────────────────────────────

export type Vars = Record<string, string | number | undefined | null>

/** Substitute `{name}` and `{name:q}` tokens.
 *  Missing variable → empty string. */
function tpl(template: string, vars: Vars): string {
  return template.replace(/\{([a-zA-Z_][\w]*)(?::(q))?\}/g, (_, name, mod) => {
    const v = vars[name]
    if (v == null) return ''
    const s = String(v)
    return mod === 'q' ? encodeURIComponent(s) : s
  })
}

function absolutise(href: string | null, base: string): string | null {
  if (!href) return null
  try { return new URL(href, base).href } catch { return null }
}

function extractVars(input: string, pattern?: string): Record<string, string> {
  if (!pattern) return {}
  try {
    return (new RegExp(pattern).exec(input)?.groups ?? {}) as Record<string, string>
  } catch { return {} }
}

// ─── HTTP layer ───────────────────────────────────────────────────

interface FetchResult {
  url:    string
  parsed: Document | unknown
}

/** Build Cookie header from user-supplied source credentials. */
function buildCookieHeader(
  manifest: SourceManifest,
  userCookies: Record<string, string>,
): string | null {
  if (!manifest.cookieNames?.length) return null
  const parts = manifest.cookieNames
    .map((name) => userCookies[name] ? `${name}=${userCookies[name]}` : null)
    .filter(Boolean)
  return parts.length > 0 ? parts.join('; ') : null
}

async function exec(
  req:         HttpRequest,
  vars:        Vars,
  userCookies: Record<string, string> = {},
  manifest?:   SourceManifest,
): Promise<FetchResult> {
  const url = tpl(req.url, vars)

  const headers: Record<string, string> = { ...(req.headers ?? {}) }

  // Inject user-supplied cookies when source requires it.
  if (manifest) {
    const cookieHdr = buildCookieHeader(manifest, userCookies)
    if (cookieHdr) headers['Cookie'] = cookieHdr
  }

  const res = await fetchSource(url, {
    headers,
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

// ─── Row abstraction ──────────────────────────────────────────────

interface Row {
  get: (selector: string) => string | null
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

// ─── Field resolution ─────────────────────────────────────────────

/** Two-pass resolve: selectors first, then `=templates` (which may
 *  reference selector results). Both passes run once — no recursion. */
function resolveFields(
  row:     Row,
  fields:  Record<string, string>,
  globals: Vars,
): Record<string, string | null> {
  const out: Record<string, string | null> = {}
  // Pass 1 — selectors
  for (const [k, decl] of Object.entries(fields)) {
    if (!decl.startsWith('=')) out[k] = row.get(decl)
  }
  // Pass 2 — templates (can reference pass-1 results via globals merge)
  for (const [k, decl] of Object.entries(fields)) {
    if (decl.startsWith('=')) {
      out[k] = tpl(decl.slice(1), { ...globals, ...out })
    }
  }
  return out
}

/** Resolve rootExtras once against the response root, returning a
 *  plain vars map. These values are available to every row's extras
 *  and fields without re-querying the root per row. */
function resolveRootExtras(
  parsed:     Document | unknown,
  mode:       'html' | 'json',
  rootExtras: Record<string, string> | undefined,
  globals:    Vars,
): Record<string, string | null> {
  if (!rootExtras) return {}
  const root = rootRow(parsed, mode)
  return resolveFields(root, rootExtras, globals)
}

// ─── Row filter (keepIf) ──────────────────────────────────────────

function passesPredicates(
  row:     Row,
  preds:   Record<string, string>,
  globals: Vars,
): boolean {
  for (const decl of Object.values(preds)) {
    const v = decl.startsWith('=')
      ? tpl(decl.slice(1), globals)
      : row.get(decl)
    if (v == null) return false
    const s = String(v).trim()
    if (!s || s === '0' || s === 'false' || s === 'null') return false
  }
  return true
}

// ─── Public API: browse ───────────────────────────────────────────

export type { BrowseArgs }

export interface ShelfDescriptor {
  id:        string
  label:     string
  hint?:     string
  paginated: boolean
}

export function getShelves(manifest: SourceManifest): ShelfDescriptor[] {
  if (isInternal(manifest)) return []
  // Adapter with fetchBrowse: expose shelves declared in manifest,
  // falling back to a synthetic "latest" shelf when none declared.
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.fetchBrowse) {
      const declared = manifest.endpoints?.shelves ?? []
      return declared.length > 0
        ? declared.map((s: ShelfManifestEntry) => ({
            id:        s.id,
            label:     s.label,
            hint:      s.hint,
            paginated: s.endpoint.pagination?.type !== 'cursor' && !!s.endpoint.pagination,
          }))
        : [{ id: 'latest', label: 'Mới nhất', paginated: true }]
    }
  }
  return (manifest.endpoints?.shelves ?? []).map((s: ShelfManifestEntry) => ({
    id:        s.id,
    label:     s.label,
    hint:      s.hint,
    paginated: !!s.endpoint.pagination,
  }))
}

export function hasSearch(manifest: SourceManifest): boolean {
  if (isInternal(manifest)) return false
  // Adapter with fetchBrowse implicitly supports search
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.fetchBrowse) return true
  }
  return !!manifest.endpoints?.search
}

export function shelfPageSize(manifest: SourceManifest, shelfId: string): number {
  if (isInternal(manifest)) return 24
  // Adapter: check manifest shelves first, fall back to PAGE_SIZE constant
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.fetchBrowse) {
      const shelf = manifest.endpoints?.shelves.find((s) => s.id === shelfId)
      const p = shelf?.endpoint.pagination
      return (p && p.type !== 'cursor') ? p.pageSize : 25
    }
  }
  const shelf = manifest.endpoints?.shelves.find((s) => s.id === shelfId)
  return shelf?.endpoint.pagination?.pageSize ?? Infinity
}

export function searchPageSize(manifest: SourceManifest): number {
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.fetchBrowse) return 25
  }
  return manifest.endpoints?.search?.pagination?.pageSize ?? Infinity
}

export async function fetchBrowse(
  manifest: SourceManifest,
  shelfId:  string | { search: true },
  args:     BrowseArgs = {},
): Promise<MangaSummary[]> {
  if (isInternal(manifest)) return []

  // Adapter override — for fully JS-rendered sites (e.g. Hitomi)
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.fetchBrowse) {
      return adapter.fetchBrowse(manifest, shelfId, args)
    }
  }

  const endpoint = typeof shelfId === 'string'
    ? manifest.endpoints?.shelves.find((s) => s.id === shelfId)?.endpoint
    : manifest.endpoints?.search
  if (!endpoint) return []
  return execBrowseEndpoint(endpoint, args, args.userCookies ?? {}, manifest)
}

async function execBrowseEndpoint(
  endpoint:    BrowseEndpoint,
  args:        BrowseArgs,
  userCookies: Record<string, string>,
  manifest:    SourceManifest,
): Promise<MangaSummary[]> {
  const page   = args.page ?? 1
  const offset = endpoint.pagination?.type === 'offset'
    ? (page - 1) * endpoint.pagination.pageSize
    : 0
  const vars: Vars = {
    q:            args.q ?? '',
    page,
    offset,
    filterParams: args.filterParams ?? '',
  }

  const { url, parsed } = await exec(endpoint, vars, userCookies, manifest)

  // rootExtras — resolved once, merged into every row's vars
  const rootE = resolveRootExtras(parsed, endpoint.parse, endpoint.rootExtras, vars)
  const rowGlobals: Vars = { ...vars, ...rootE }

  const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)
  const out: MangaSummary[] = []

  for (const row of rows) {
    // Optional row filter
    if (endpoint.keepIf && !passesPredicates(row, endpoint.keepIf, rowGlobals)) continue

    const extras = endpoint.extras
      ? resolveFields(row, endpoint.extras, rowGlobals)
      : {}
    const f = resolveFields(
      row,
      endpoint.fields as Record<string, string>,
      { ...rowGlobals, ...extras },
    )
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

// ─── Public API: manga detail ─────────────────────────────────────

export async function fetchMangaDetail(
  manifest: SourceManifest,
  mangaUrl: string,
  args:     { language?: string; userCookies?: Record<string, string> } = {},
): Promise<MangaDetail> {
  if (isInternal(manifest) || !manifest.endpoints?.manga) {
    throw new Error(`Source ${manifest.id} has no manga endpoint`)
  }

  const userCookies = args.userCookies ?? {}

  // Adapter override for manga detail (optional — most adapters only
  // override chapter pages).
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.fetchMangaDetail) {
      return adapter.fetchMangaDetail(manifest, mangaUrl, userCookies)
    }
  }
  const endpoint = manifest.endpoints.manga
  const vars: Vars  = {
    mangaUrl,
    language: args.language ?? manifest.languages[0],
    ...extractVars(mangaUrl, endpoint.extract),
  }

  const { url: baseUrl, parsed } = await exec(endpoint, vars, userCookies, manifest)

  // rootExtras (new) + legacy extras — both resolved against root
  const rootE = resolveRootExtras(parsed, endpoint.parse, endpoint.rootExtras, vars)
  const root  = rootRow(parsed, endpoint.parse)
  const extras = endpoint.extras
    ? resolveFields(root, endpoint.extras, { ...vars, ...rootE })
    : {}
  const allExtras: Vars = { ...vars, ...rootE, ...extras }
  const f = resolveFields(root, endpoint.fields as Record<string, string>, allExtras)

  // Chapters: inline or external
  let chapters: MangaChapterRef[] = []
  if (endpoint.chapters) {
    chapters = collectChapters(
      parsed, endpoint.parse, endpoint.chapters,
      baseUrl, allExtras, manifest,
    )
  } else if (manifest.endpoints?.chaptersApi) {
    chapters = await fetchChaptersExternal(
      manifest.endpoints.chaptersApi,
      allExtras,
      manifest,
      userCookies,
    )
  }

  // Stamp manga-level updatedAt onto latest chapter when source
  // doesn't report per-chapter dates (OTruyen pattern).
  const mangaUpdatedAt = f.updatedAt ?? null
  if (mangaUpdatedAt && chapters.length > 0 && chapters.every((c) => !c.date)) {
    const latest = pickLatestChapter(chapters)
    if (latest) latest.date = mangaUpdatedAt
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

// ─── Chapter helpers ──────────────────────────────────────────────

function collectChapters(
  parsed:    unknown,
  parseMode: 'html' | 'json',
  spec:      ChapterListSpec,
  baseUrl:   string,
  globals:   Vars,
  manifest:  SourceManifest,
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
  endpoint:    ChaptersApiEndpoint,
  vars:        Vars,
  manifest:    SourceManifest,
  userCookies: Record<string, string> = {},
): Promise<MangaChapterRef[]> {
  const norm = compileChapterNumberNorm(
    endpoint.chapterNumberNorm ?? manifest.chapterNumberNorm,
  )
  const pageSize = endpoint.pagination?.pageSize

  const buildRows = (rows: Row[], baseUrl: string, globals: Vars): MangaChapterRef[] => {
    const kept = endpoint.keepIf
      ? rows.filter((r) => passesPredicates(r, endpoint.keepIf!, globals))
      : rows
    return kept
      .map((r) => {
        // Resolve per-row extras first so = templates in fields can use them.
        const rowExtras = endpoint.extras
          ? resolveFields(r, endpoint.extras, globals)
          : {}
        return buildChapter(r, endpoint.fields, baseUrl, { ...globals, ...rowExtras }, norm)
      })
      .filter((c): c is MangaChapterRef => c !== null)
  }

  // No pagination declared — single fetch (original behaviour).
  if (!pageSize) {
    const { url, parsed } = await exec(endpoint, { ...vars, page: 1, offset: 0 }, userCookies, manifest)
    const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)
    return buildRows(rows, url, vars)
  }

  // Paginated — fetch until a page returns fewer rows than pageSize.
  const all: MangaChapterRef[] = []
  let page   = 1
  let offset = 0
  while (true) {
    const { url, parsed } = await exec(
      endpoint,
      { ...vars, page, offset },
      userCookies,
      manifest,
    )
    const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)
    all.push(...buildRows(rows, url, vars))
    if (rows.length < pageSize) break
    page++
    offset += pageSize
  }
  return all
}

function buildChapter(
  row:     Row,
  fields:  ChapterFields,
  baseUrl: string,
  globals: Vars,
  norm:    ReturnType<typeof compileChapterNumberNorm>,
): MangaChapterRef | null {
  const f      = resolveFields(row, fields as unknown as Record<string, string>, globals)
  const url    = absolutise(f.url, baseUrl)
  if (!url) return null
  const number = f.number ?? ''
  const title  = f.title  ?? null
  const label  = f.label  ?? composeChapterLabel(number, title)
  return {
    id:         url,
    url,
    number,
    numberNorm: applyChapterNumberNorm(norm, { number, label }),
    title,
    label,
    date:       f.date      ?? null,
    language:   f.language  ?? null,
    scanlator:  f.scanlator ?? null,
  }
}

function composeChapterLabel(number: string, title: string | null): string {
  const n = number.trim()
  const t = title?.trim() ?? ''
  if (n && t) return `Chương ${n} · ${t}`
  if (n)      return `Chương ${n}`
  if (t)      return t
  return '?'
}

function pickLatestChapter(chapters: MangaChapterRef[]): MangaChapterRef | null {
  let best: MangaChapterRef | null = null
  let bestKey = -Infinity
  for (const c of chapters) {
    const k = parseFloat(c.numberNorm)
    if (Number.isFinite(k) && k > bestKey) { best = c; bestKey = k }
  }
  return best ?? chapters[chapters.length - 1] ?? null
}

// ─── Public API: chapter pages ────────────────────────────────────

export async function fetchChapterPages(
  manifest:    SourceManifest,
  chapterUrl:  string,
  userCookies: Record<string, string> = {},
): Promise<ChapterPages> {
  if (isInternal(manifest)) {
    throw new Error(`Source ${manifest.id} has no chapter endpoint`)
  }

  // Delegate to adapter when declared — adapter handles all imperative
  // logic (per-page ext maps, JS-rendered URLs, CDN selection, etc.).
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (!adapter) throw new Error(`Unknown adapter: "${manifest.adapter}"`)
    return adapter.fetchChapterPages(manifest, chapterUrl, userCookies)
  }

  if (!manifest.endpoints?.chapter) {
    throw new Error(`Source ${manifest.id} has no chapter endpoint`)
  }
  const endpoint = manifest.endpoints.chapter
  const vars: Vars = {
    chapterUrl,
    ...extractVars(chapterUrl, endpoint.extract),
  }
  const { url: requestUrl, parsed } = await exec(endpoint, vars, userCookies, manifest)

  // Shape 1: list of rows each carrying a URL.
  if (endpoint.list && endpoint.fields) {
    const rows = rowsFrom(parsed, endpoint.list, endpoint.parse)
    const pages: string[] = []
    for (const r of rows) {
      const f = resolveFields(r, endpoint.fields as unknown as Record<string, string>, vars)
      const u = absolutise(f.url, requestUrl)
      if (u) pages.push(u)
    }
    return { url: chapterUrl, pages }
  }

  // Shape 2 & 3: pages spec (iterate array OR count range).
  if (endpoint.pages) {
    const spec = endpoint.pages

    // rootExtras for chapter endpoint (e.g. CDN host from hidden input)
    const rootE = resolveRootExtras(parsed, endpoint.parse, endpoint.rootExtras, vars)
    const root  = rootRow(parsed, endpoint.parse)

    // Resolve all extras as scalars first.
    const scalarExtras: Record<string, string | null> = { ...rootE }
    const iterKey   = spec.iterate
    const countKey  = spec.count
    const skipInExtras = iterKey ?? countKey

    for (const [k, sel] of Object.entries(spec.extras)) {
      if (k === skipInExtras) continue
      scalarExtras[k] = root.get(sel)
    }

    const allVars: Vars = { ...vars, ...scalarExtras }
    const pages: string[] = []

    if (iterKey) {
      // Shape 2: iterate over an array field.
      const filesSel = spec.extras[iterKey]
      if (!filesSel) throw new Error(`pages.iterate "${iterKey}" not in extras`)

      const files = endpoint.parse === 'json'
        ? queryJsonAll(parsed, filesSel)
        : queryHtmlAll(parsed as Document, filesSel).map((el) => el.textContent)

      for (const file of files) {
        const f = typeof file === 'string' ? file : String(file ?? '')
        if (!f) continue
        pages.push(tpl(spec.template, { ...allVars, file: f }))
      }

    } else if (countKey) {
      // Shape 3: numeric range 1..N.
      const countSel = spec.extras[countKey]
      if (!countSel) throw new Error(`pages.count "${countKey}" not in extras`)

      const rawCount = root.get(countSel)
      const n = parseInt(rawCount ?? '', 10)
      if (!Number.isFinite(n) || n <= 0) {
        throw new Error(`pages.count resolved to invalid number: "${rawCount}"`)
      }
      for (let i = 1; i <= n; i++) {
        pages.push(tpl(spec.template, { ...allVars, file: String(i) }))
      }

    } else {
      throw new Error(`pages spec needs either "iterate" or "count" (${manifest.id})`)
    }

    return { url: chapterUrl, pages }
  }

  throw new Error(`chapter endpoint missing list/pages spec (${manifest.id})`)
}

// ─── Lazy page URL resolution ─────────────────────────────────────

export async function resolvePageUrl(
  manifest:    SourceManifest,
  token:       string,
  userCookies: Record<string, string> = {},
): Promise<string> {
  if (manifest.adapter) {
    const adapter = getAdapter(manifest.adapter)
    if (adapter?.resolvePageUrl) {
      return adapter.resolvePageUrl(manifest, token, userCookies)
    }
  }
  throw new Error(
    `resolvePageUrl called on source "${manifest.id}" which has no adapter.resolvePageUrl`,
  )
}

// ─── Filter param assembly ────────────────────────────────────────

export function getFilters(manifest: SourceManifest) {
  return manifest.filters ?? []
}

export function getDefaultFilterState(
  manifest: SourceManifest,
): Record<string, string | string[]> {
  return { ...(manifest.defaults ?? {}) }
}

export function assembleFilterParams(
  manifest: SourceManifest,
  state:    Record<string, string | string[]>,
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

/** Build a typed filter state for adapters. Maps each active option's
 *  `FilterOption.value ?? FilterOption.id` so adapters get structured
 *  values without re-parsing the assembled `filterParams` string. */
export function assembleFilterState(
  manifest: SourceManifest,
  state:    Record<string, string | string[]>,
): Record<string, string | string[]> {
  if (!manifest.filters) return {}
  const out: Record<string, string | string[]> = {}
  for (const filter of manifest.filters) {
    const sel = state[filter.id]
    if (sel == null) continue
    const ids = Array.isArray(sel) ? sel : [sel]
    const values = ids
      .map((id) => {
        const opt = filter.options.find((o) => o.id === id)
        return opt ? (opt.value ?? opt.id) : null
      })
      .filter((v): v is string => v !== null)
    if (values.length === 0) continue
    out[filter.id] = filter.type === 'select' ? (values[0]!) : values
  }
  return out
}

// ─── Lang list helper ─────────────────────────────────────────────

function parseLangList(raw: string | null | undefined): string[] | null {
  if (!raw) return null
  if (raw.startsWith('[')) {
    try {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        return parsed.filter((x): x is string => typeof x === 'string')
      }
    } catch { /* fall through */ }
  }
  const parts = raw.split(/[\s,]+/).filter((s) => s.length > 0)
  return parts.length > 0 ? parts : null
}
