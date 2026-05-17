// Hitomi.la adapter.
//
// Why adapter instead of declarative manifest:
//   - The site is fully JS-rendered — HTML pages contain no gallery data.
//   - Listing uses a binary nozomi index (packed big-endian int32 gallery IDs)
//     fetched with HTTP Range headers for pagination.
//   - Image URLs are computed from per-file hashes via gg.js CDN routing logic
//     (switch-case subdomain selector + base path that rotates daily).
//
// Browse pipeline:
//   1. Fetch nozomi index → slice page → list of gallery IDs (fast).
//   2. Return token stubs immediately — no galleryblock fetches here.
//   3. Each card calls resolveGalleryBlock on viewport entry → fetches
//      galleryblock/{id}.html, cached forever (immutable per gallery).
//
// Result: grid renders instantly; titles/covers stream in as cards scroll
// into view. React Query caches each block by ID so page flips and shelf
// switches never re-fetch already-seen galleries.
//
//   fetchMangaDetail:
//     1. Extract gallery ID from URL.
//     2. Fetch galleries/{id}.js → parse `var galleryinfo = {...}`.
//     3. Return metadata + single-chapter stub.
//
//   fetchChapterPages:
//     1. Fetch galleries/{id}.js for file list + hashes.
//     2. Fetch gg.js for current CDN routing.
//     3. Compute webp image URL for every file.

import { pfetch } from '../proxy'
import type { BrowseArgs, ChapterPages, MangaDetail, MangaSummary, SourceManifest } from '../manifest/types'
import type { SourceAdapter } from './types'

// ── constants ────────────────────────────────────────────────────────────────

const DOMAIN2   = 'gold-usergeneratedcontent.net'
const LTN       = `https://ltn.${DOMAIN2}`
const PAGE_SIZE = 25

// ── gg.js CDN routing ────────────────────────────────────────────────────────
// gg.js exports: { m(g): 0|1, s(hash): string, b: string }
// We parse it once and cache for the session (it rotates ~daily).

interface GgData {
  /** Switch-case result for a numeric g value. */
  m:    (g: number) => number
  /** Extracts the subdirectory number from a hash. */
  s:    (hash: string) => string
  /** Base path prefix, e.g. "1778860802/" */
  b:    string
  /** Timestamp of last fetch (ms). */
  fetchedAt: number
}

let ggCache: GgData | null = null
const GG_TTL = 60 * 60 * 1000  // 1 hour

async function fetchGg(): Promise<GgData> {
  const now = Date.now()
  if (ggCache && now - ggCache.fetchedAt < GG_TTL) return ggCache

  const res = await pfetch(`${LTN}/gg.js`, {
    headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
  })
  if (!res.ok) throw new Error(`hitomi gg.js: HTTP ${res.status}`)
  const text = await res.text()

  // Parse `b: '...'`
  const bMatch = /\bb:\s*'([^']+)'/.exec(text)
  const b = bMatch?.[1] ?? ''

  // Parse `m: function(g) { var o = 0; switch(g) { case X: case Y: ... o = 1; } return o; }`
  // Extract all case values that produce o=1 (non-default branch sets o=1)
  // Strategy: find the switch body, extract case values before any `o = 1`
  const set1 = new Set<number>()
  const switchBody = /switch\s*\(g\)\s*\{([\s\S]+?)\}\s*return/.exec(text)?.[1] ?? ''
  // Each segment ends in "o = 1" or "o = 0"; split on "o ="
  const segments = switchBody.split(/o\s*=\s*(\d)/)
  // segments: [caseBlock, value, caseBlock, value, ...]
  for (let i = 0; i < segments.length - 1; i += 2) {
    const val = parseInt(segments[i + 1]!, 10)
    if (val !== 1) continue
    const block = segments[i]!
    const cases = block.match(/case\s+(\d+):/g) ?? []
    for (const c of cases) {
      const n = parseInt(c.replace(/case\s+/, '').replace(':', ''), 10)
      set1.add(n)
    }
  }

  const m = (g: number) => (set1.has(g) ? 1 : 0)

  // s(hash): same logic as hitomi's gg.s
  const s = (hash: string): string => {
    const match = /(..)(.)$/.exec(hash)
    if (!match) return '0'
    return parseInt(match[2]! + match[1]!, 16).toString(10)
  }

  ggCache = { m, s, b, fetchedAt: now }
  return ggCache
}

// ── image URL construction ───────────────────────────────────────────────────

// Matches hitomi's subdomain_from_url:
//   dir='webp' → prefix 'w' + (1 + gg.m(g))
//   dir='avif' → prefix 'a' + (1 + gg.m(g))
function subdomainFromHash(hash: string, ext: 'webp' | 'avif', gg: GgData): string {
  const match = /([0-9a-f]{2})([0-9a-f])$/.exec(hash)
  if (!match) return ext === 'avif' ? 'a1' : 'w1'
  const g      = parseInt(match[2]! + match[1]!, 16)
  const prefix = ext === 'avif' ? 'a' : 'w'
  return prefix + (1 + gg.m(g))
}

function imageUrl(hash: string, ext: 'webp' | 'avif', gg: GgData): string {
  const sub      = subdomainFromHash(hash, ext, gg)
  const fullPath = gg.b + gg.s(hash) + '/' + hash
  return `https://${sub}.${DOMAIN2}/${fullPath}.${ext}`
}

// ── helpers ──────────────────────────────────────────────────────────────────

const GALLERY_ID_RE = /-(\d+)\.html/

function extractGalleryId(url: string): string | null {
  return GALLERY_ID_RE.exec(url)?.[1] ?? null
}

// ── nozomi index ─────────────────────────────────────────────────────────────
// Hitomi's own JS fetches nozomi files in full (no Range). The bunle-cdn
// proxy does not reliably forward Range from X-Proxy-Headers (Cloudflare
// edge cache truncates partial responses). We fetch the full file once,
// cache it per URL, then slice client-side for each page request.
//
// Cache design:
//   - Keyed by nozomi URL (one entry per type/language combination).
//   - Each URL is a Promise<ArrayBuffer> so concurrent calls for the
//     same URL share one inflight fetch — no duplicate downloads.
//   - TTL 10 min: nozomi updates ~hourly with new galleries; 10 min is
//     fresh enough without re-downloading 4MB on every page flip.
//   - index-all (4.5 MB) is expensive; filtered indices are smaller
//     (doujinshi-all ~2.5 MB, per-language ~0.6 MB). The UI filter
//     chips naturally push users toward smaller indices.

interface NozomiEntry {
  promise:   Promise<ArrayBuffer>
  fetchedAt: number
}

const nozomiCache = new Map<string, NozomiEntry>()
const NOZOMI_TTL  = 10 * 60_000  // 10 minutes

function getNozomiBuffer(nozomiUrl: string): Promise<ArrayBuffer> {
  const now     = Date.now()
  const cached  = nozomiCache.get(nozomiUrl)
  if (cached && now - cached.fetchedAt < NOZOMI_TTL) return cached.promise

  const promise = pfetch(nozomiUrl, {
    headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
  }).then((res) => {
    if (!res.ok) throw new Error(`hitomi nozomi: HTTP ${res.status}`)
    return res.arrayBuffer()
  })
  nozomiCache.set(nozomiUrl, { promise, fetchedAt: now })
  return promise
}

async function fetchNozomiPage(
  nozomiUrl: string,
  page:      number,   // 0-indexed
): Promise<number[]> {
  const buf   = await getNozomiBuffer(nozomiUrl)
  const start = page * PAGE_SIZE * 4
  const end   = start + PAGE_SIZE * 4
  const slice = buf.slice(start, end)
  const view  = new DataView(slice)
  const ids: number[] = []
  for (let i = 0; i + 3 < slice.byteLength; i += 4) {
    ids.push(view.getInt32(i, false))  // big-endian
  }
  return ids
}

// ── galleryblock parsing ──────────────────────────────────────────────────────

interface GalleryBlock {
  id:    number
  url:   string
  title: string
  cover: string | null
}

// Gallery blocks are immutable once published — cache forever (session-scoped).
// Concurrent requests for the same id share one inflight Promise.
const galleryBlockCache = new Map<number, Promise<GalleryBlock>>()

function fetchGalleryBlock(id: number): Promise<GalleryBlock> {
  if (galleryBlockCache.has(id)) return galleryBlockCache.get(id)!

  const promise = pfetch(`${LTN}/galleryblock/${id}.html`, {
    headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
  }).then(async (res) => {
    if (!res.ok) throw new Error(`hitomi galleryblock ${id}: HTTP ${res.status}`)
    const doc   = new DOMParser().parseFromString(await res.text(), 'text/html')
    const href  = doc.querySelector('h1.lillie a')?.getAttribute('href') ?? ''
    const title = doc.querySelector('h1.lillie a')?.textContent?.trim() ?? `#${id}`
    const img   = doc.querySelector('a.lillie img')
    const cover = img?.getAttribute('data-src') ?? img?.getAttribute('src') ?? null
    return {
      id,
      url:   href ? `https://hitomi.la${href}` : `https://hitomi.la/galleries/${id}.html`,
      title,
      cover: cover ? (cover.startsWith('//') ? 'https:' + cover : cover) : null,
    } satisfies GalleryBlock
  })

  galleryBlockCache.set(id, promise)
  return promise
}

// ── galleryinfo ──────────────────────────────────────────────────────────────

interface GalleryFile {
  name:    string
  hash:    string
  hasavif?: number
  haswebp?: number
}

interface GalleryInfo {
  title:         string | null
  japanese_title: string | null
  galleryurl:    string
  language:      string | null
  type:          string | null
  artists:       Array<{ artist: string }> | null
  tags:          Array<{ tag: string; female?: string; male?: string }> | null
  files:         GalleryFile[]
  datepublished: string | null
}

async function fetchGalleryInfo(id: string): Promise<GalleryInfo> {
  const res = await pfetch(`${LTN}/galleries/${id}.js`, {
    headers: { Referer: 'https://hitomi.la/', Origin: 'https://hitomi.la' },
  })
  if (!res.ok) throw new Error(`hitomi galleries/${id}.js: HTTP ${res.status}`)
  const text = await res.text()
  // Strip `var galleryinfo = ` prefix
  const json = text.replace(/^var galleryinfo\s*=\s*/, '').replace(/;\s*$/, '')
  return JSON.parse(json) as GalleryInfo
}

// ── nozomi URL from typed filter state ───────────────────────────────────────
// Hitomi nozomi URL scheme (from search.js `nozomi_address_from_state`):
//   all types, all langs:   /n/index-all.nozomi
//   all types, one lang:    /n/index-{lang}.nozomi
//   one type,  all langs:   /n/type/{type}-all.nozomi
//   one type,  one lang:    /n/type/{type}-{lang}.nozomi
// Types: doujinshi, manga, artistcg, gamecg, imageset, anime
// Languages: all, japanese, chinese, english, korean, …

function nozomiUrlFromFilterState(
  state: Record<string, string | string[]> | undefined,
): string {
  const get = (key: string): string | null => {
    const v = state?.[key]
    return (typeof v === 'string' && v !== 'all') ? v : null
  }
  const type = get('type')
  const lang = get('language')

  if (type && lang) return `${LTN}/n/type/${encodeURIComponent(type)}-${encodeURIComponent(lang)}.nozomi`
  if (type)         return `${LTN}/n/type/${encodeURIComponent(type)}-all.nozomi`
  if (lang)         return `${LTN}/n/index-${encodeURIComponent(lang)}.nozomi`
  return `${LTN}/n/index-all.nozomi`
}

// ── adapter ──────────────────────────────────────────────────────────────────

export const hitomiAdapter: SourceAdapter = {
  async fetchBrowse(
    _manifest:     SourceManifest,
    shelfOrSearch: string | { search: true },
    args: BrowseArgs = {},
  ): Promise<MangaSummary[]> {
    const page = Math.max(0, (args.page ?? 1) - 1)

    let nozomiUrl: string
    if (typeof shelfOrSearch !== 'string' && args.q?.trim()) {
      const term = args.q.trim().toLowerCase().replace(/\s+/g, '_')
      nozomiUrl = `${LTN}/n/${encodeURIComponent(term)}-all.nozomi`
    } else {
      nozomiUrl = nozomiUrlFromFilterState(args.filterState)
    }

    const ids = await fetchNozomiPage(nozomiUrl, page)
    if (!ids.length) return []

    // Prefetch next page into nozomi cache while current page renders.
    // nozomi buffer is already in cache; slicing is synchronous — this
    // just ensures the ArrayBuffer is warm before user scrolls.
    void fetchNozomiPage(nozomiUrl, page + 1).catch(() => {})

    // Fetch all 25 gallery blocks in parallel. galleryBlockCache ensures
    // re-visits and page flips never re-fetch already-seen galleries.
    const blocks = await Promise.allSettled(ids.map(fetchGalleryBlock))
    return blocks
      .filter((r): r is PromiseFulfilledResult<GalleryBlock> => r.status === 'fulfilled')
      .map(({ value: b }) => ({
        id:    b.url,
        url:   b.url,
        title: b.title,
        cover: b.cover,
      }))
  },

  async fetchMangaDetail(
    _manifest:   SourceManifest,
    mangaUrl:    string,
    _userCookies: Record<string, string>,
  ): Promise<MangaDetail> {
    const id = extractGalleryId(mangaUrl)
    if (!id) throw new Error(`hitomi: cannot extract gallery ID from: ${mangaUrl}`)

    const info  = await fetchGalleryInfo(id)
    const title = info.title ?? info.japanese_title ?? `Gallery #${id}`
    const block = await fetchGalleryBlock(parseInt(id, 10))

    const langMap: Record<string, string> = {
      japanese: 'ja', chinese: 'zh', english: 'en', korean: 'ko',
    }
    const lang = info.language ? (langMap[info.language] ?? info.language) : null

    const date = info.datepublished
      ? info.datepublished.slice(0, 10)
      : null

    return {
      id:          mangaUrl,
      url:         mangaUrl,
      title,
      cover:       block.cover,
      description: info.type ?? null,
      author:      info.artists?.map((a) => a.artist).join(', ') ?? null,
      status:      info.language ?? null,
      availableLanguages: lang ? [lang] : null,
      chapters: [{
        id:         mangaUrl,
        url:        mangaUrl,
        number:     '1',
        numberNorm: '1',
        label:      `${info.files.length} pages`,
        title:      null,
        date,
        language:   lang,
        scanlator:  null,
      }],
    }
  },

  async fetchChapterPages(
    _manifest:    SourceManifest,
    chapterUrl:  string,
    _userCookies: Record<string, string>,
  ): Promise<ChapterPages> {
    const id = extractGalleryId(chapterUrl)
    if (!id) throw new Error(`hitomi: cannot extract gallery ID from: ${chapterUrl}`)

    const [info, gg] = await Promise.all([
      fetchGalleryInfo(id),
      fetchGg(),
    ])

    const pages = info.files.map((file) => {
      const ext: 'avif' | 'webp' = file.hasavif ? 'avif' : 'webp'
      return imageUrl(file.hash, ext, gg)
    })

    return { url: chapterUrl, pages }
  },
}
