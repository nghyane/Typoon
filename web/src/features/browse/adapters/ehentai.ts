// E-Hentai adapter.
//
// Why adapter instead of declarative manifest:
//   - Image URLs are signed Hath-network CDN tokens obtained via showpage
//     API — one call per page, cannot be expressed declaratively.
//
// Token format (opaque to reader):
//   "{gid}/{page}/{galleryToken}/{showkey}"
//   All fields the resolver needs are in the token; no shared state.
//
// Performance design:
//
//   fetchChapterPages  — 3 network calls total:
//     1. gdata API        → pageCount (to create N slots)
//     2. Thumb page ?p=0  → hashes for pages 1–20
//     3. Reader page 1    → showkey + URL for page 1 (put directly in pages[0])
//     Returns immediately. Reader shows N slots; page 1 already has its URL.
//
//   resolvePageUrl  — called per-page on viewport entry (React Query cached):
//     - Needs hash_{page-1} to call showpage(page-1) → URL for current page.
//     - Hash is looked up from a module-level thumb-page cache keyed by
//       (gid, thumbPageIndex). fetchChapterPages pre-warms thumb page 0.
//       Further thumb pages are fetched on-demand (one fetch per 20 pages).
//     - staleTime:Infinity in React Query → resolved URL cached for session.
//
//   Thumb-page cache:
//     Module-level Map<"gid/p", Promise<Map<page,hash>>>. Two resolvePageUrl
//     calls for pages on the same thumb page share one inflight Promise —
//     no duplicate fetches even when multiple pages enter the viewport at once.

import { pfetch } from '../proxy'
import { queryHtmlAll } from '../manifest/selectors'
import type { ChapterPages, MangaDetail, SourceManifest } from '../manifest/types'
import type { SourceAdapter } from './types'

// ── constants ────────────────────────────────────────────────────────────────

const API             = 'https://api.e-hentai.org/api.php'
const GALLERY_RE      = /e-hentai\.org\/g\/(\d+)\/([a-f0-9]+)/
const THUMBS_PER_PAGE = 20
const PAGE_HASH_RE    = /\/s\/([a-f0-9]+)\/\d+-(\d+)/

// ── thumb-page promise cache ─────────────────────────────────────────────────
// Key: "${gid}/${thumbPageIndex}"  Value: resolving Promise<Map<pageNum, hash>>
// Multiple concurrent resolvePageUrl calls for the same thumb page share the
// same Promise — no duplicate network requests.

const thumbCache = new Map<string, Promise<Map<number, string>>>()

function thumbCacheKey(gid: string, thumbPageIdx: number): string {
  return `${gid}/${thumbPageIdx}`
}

function thumbPageIdxFor(pageNum: number): number {
  return Math.floor((pageNum - 1) / THUMBS_PER_PAGE)
}

// ── helpers ──────────────────────────────────────────────────────────────────

function parseGalleryUrl(url: string): { gid: string; token: string } | null {
  const m = GALLERY_RE.exec(url)
  if (!m) return null
  return { gid: m[1]!, token: m[2]! }
}

function cookieHeader(userCookies: Record<string, string>): string | null {
  const entries = Object.entries(userCookies)
  return entries.length ? entries.map(([k, v]) => `${k}=${v}`).join('; ') : null
}

async function fetchHtml(
  url: string,
  userCookies: Record<string, string>,
): Promise<Document> {
  const headers: Record<string, string> = { Referer: 'https://e-hentai.org/' }
  const c = cookieHeader(userCookies)
  if (c) headers['Cookie'] = c
  const res = await pfetch(url, { headers })
  if (!res.ok) throw new Error(`E-Hentai: HTTP ${res.status} on ${url}`)
  return new DOMParser().parseFromString(await res.text(), 'text/html')
}

async function postApi(
  body: object,
  userCookies: Record<string, string>,
): Promise<Record<string, unknown>> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const c = cookieHeader(userCookies)
  if (c) headers['Cookie'] = c
  const res = await pfetch(API, {
    headers,
    init: { method: 'POST', body: JSON.stringify(body) },
  })
  if (!res.ok) throw new Error(`E-Hentai API: HTTP ${res.status}`)
  return res.json() as Promise<Record<string, unknown>>
}

// ── gdata ────────────────────────────────────────────────────────────────────

async function fetchGdata(
  gid: string,
  token: string,
  userCookies: Record<string, string>,
): Promise<Record<string, unknown>> {
  const json = await postApi(
    { method: 'gdata', gidlist: [[parseInt(gid, 10), token]], namespace: 1 },
    userCookies,
  )
  const meta = (json as { gmetadata?: unknown[] }).gmetadata?.[0]
  if (!meta || typeof meta !== 'object') throw new Error('E-Hentai gdata: no metadata')
  return meta as Record<string, unknown>
}

// ── thumb page fetching ───────────────────────────────────────────────────────

function parseThumbPage(doc: Document): Map<number, string> {
  const map = new Map<number, string>()
  for (const el of queryHtmlAll(doc, '#gdt a')) {
    const m = PAGE_HASH_RE.exec(el.getAttribute('href') ?? '')
    if (m) map.set(parseInt(m[2]!, 10), m[1]!)
  }
  return map
}

/** Fetch one gallery thumb page and populate cache. Returns hash map for that
 *  thumb page. Concurrent calls for the same key share one in-flight Promise. */
function getThumbPage(
  gid: string,
  galleryToken: string,
  thumbPageIdx: number,
  userCookies: Record<string, string>,
): Promise<Map<number, string>> {
  const key = thumbCacheKey(gid, thumbPageIdx)
  if (!thumbCache.has(key)) {
    const p = fetchHtml(
      `https://e-hentai.org/g/${gid}/${galleryToken}/?p=${thumbPageIdx}`,
      userCookies,
    ).then(parseThumbPage)
    thumbCache.set(key, p)
  }
  return thumbCache.get(key)!
}

/** Look up hash for a specific page number. Fetches the required thumb page
 *  lazily (shared Promise so parallel calls don't duplicate). */
async function getHash(
  gid: string,
  galleryToken: string,
  pageNum: number,
  userCookies: Record<string, string>,
): Promise<string | null> {
  const thumbPageIdx = thumbPageIdxFor(pageNum)
  const map = await getThumbPage(gid, galleryToken, thumbPageIdx, userCookies)
  return map.get(pageNum) ?? null
}

// ── showpage → CDN URL ────────────────────────────────────────────────────────

/** Resolve URL for page `page` (1-indexed).
 *  Page 1: parse #img src from reader HTML.
 *  Page N>1: call showpage(N-1, imgkey=hash_{N-1}) → parse CDN URL from i3. */
async function resolveUrl(
  gid: string,
  page: number,
  galleryToken: string,
  showkey: string,
  userCookies: Record<string, string>,
): Promise<string> {
  if (page === 1) {
    const hash = await getHash(gid, galleryToken, 1, userCookies)
    if (!hash) throw new Error(`E-Hentai: hash missing for page 1 (gid ${gid})`)
    const doc = await fetchHtml(`https://e-hentai.org/s/${hash}/${gid}-1`, userCookies)
    const src = (doc.querySelector('#img') as HTMLImageElement | null)?.src
    if (!src) throw new Error(`E-Hentai: #img not found on reader page 1 (gid ${gid})`)
    return src
  }

  const prevHash = await getHash(gid, galleryToken, page - 1, userCookies)
  if (!prevHash) throw new Error(`E-Hentai: hash missing for page ${page - 1} (gid ${gid})`)

  const json   = await postApi(
    { method: 'showpage', gid: parseInt(gid, 10), page: page - 1, imgkey: prevHash, showkey },
    userCookies,
  )
  const i3Html = (json['i3'] as string | undefined) ?? ''
  const m      = /id="img"[^>]*src="([^"]+)"/.exec(i3Html)
  if (!m?.[1]) throw new Error(`E-Hentai showpage: no img in i3 for gid ${gid} page ${page}`)
  return m[1]
}

// ── token codec ───────────────────────────────────────────────────────────────
// Encodes everything resolvePageUrl needs; no shared mutable state in token.

function encodeToken(gid: string, page: number, galleryToken: string, showkey: string): string {
  return `${gid}\x00${page}\x00${galleryToken}\x00${showkey}`
}

function decodeToken(token: string): {
  gid: string; page: number; galleryToken: string; showkey: string
} | null {
  const parts = token.split('\x00')
  if (parts.length !== 4) return null
  const page = parseInt(parts[1]!, 10)
  if (!Number.isFinite(page)) return null
  return { gid: parts[0]!, page, galleryToken: parts[2]!, showkey: parts[3]! }
}

// ── adapter ───────────────────────────────────────────────────────────────────

export const ehentaiAdapter: SourceAdapter = {
  async fetchMangaDetail(
    _manifest:   SourceManifest,
    mangaUrl:    string,
    userCookies: Record<string, string>,
  ): Promise<MangaDetail> {
    const ids = parseGalleryUrl(mangaUrl)
    if (!ids) throw new Error(`E-Hentai: cannot parse URL: ${mangaUrl}`)

    const meta     = await fetchGdata(ids.gid, ids.token, userCookies)
    const title    = (meta.title as string | undefined)
                  ?? (meta.title_jpn as string | undefined)
                  ?? '(không tên)'
    const tags     = (meta.tags as string[] | undefined) ?? []
    const langTag  = tags.find((t) => t.startsWith('language:'))
    const lang     = langTag ? langTag.replace('language:', '') : null
    const pageCount = parseInt(String(meta.filecount ?? '0'), 10)
    const date     = meta.posted
      ? new Date(parseInt(String(meta.posted), 10) * 1000).toISOString().slice(0, 10)
      : null

    return {
      id: mangaUrl, url: mangaUrl,
      title,
      cover:       (meta.thumb as string | undefined) ?? null,
      description: (meta.category as string | undefined) ?? null,
      author:      (meta.uploader as string | undefined) ?? null,
      status:      lang ? `language:${lang}` : null,
      availableLanguages: lang ? [lang] : null,
      chapters: [{
        id: mangaUrl, url: mangaUrl,
        number: '1', numberNorm: '1',
        label:     pageCount > 0 ? `${pageCount} pages` : 'Ch.1',
        title:     null, date, language: lang,
        scanlator: (meta.uploader as string | undefined) ?? null,
      }],
    }
  },

  /** 3 network calls: gdata + thumb page 0 + reader page 1.
   *  pages[0] is populated immediately; all other pages resolve lazily. */
  async fetchChapterPages(
    _manifest:   SourceManifest,
    chapterUrl:  string,
    userCookies: Record<string, string>,
  ): Promise<ChapterPages> {
    const ids = parseGalleryUrl(chapterUrl)
    if (!ids) throw new Error(`E-Hentai: cannot parse URL: ${chapterUrl}`)

    const { gid, token: galleryToken } = ids

    // 1. gdata → pageCount  (parallel with nothing — needed first)
    const meta      = await fetchGdata(gid, galleryToken, userCookies)
    const pageCount = parseInt(String(meta.filecount ?? '0'), 10)
    if (!pageCount) throw new Error(`E-Hentai: filecount 0 for ${chapterUrl}`)

    // 2. Thumb page 0 + reader page 1 in parallel.
    //    Thumb page 0 warms the cache for pages 1–20.
    //    Reader page 1 gives showkey + immediate URL for page 1.
    const thumbP  = getThumbPage(gid, galleryToken, 0, userCookies)   // warms cache
    const readerP = thumbP.then(async (thumbMap) => {
      const hash1 = thumbMap.get(1)
      if (!hash1) throw new Error(`E-Hentai: hash missing for page 1`)
      return fetchHtml(`https://e-hentai.org/s/${hash1}/${gid}-1`, userCookies)
    })

    const [, readerDoc] = await Promise.all([thumbP, readerP])

    // Extract showkey
    let showkey = ''
    for (const script of Array.from(readerDoc.querySelectorAll('script'))) {
      const m = /var showkey="([^"]+)"/.exec(script.textContent ?? '')
      if (m) { showkey = m[1]!; break }
    }
    if (!showkey) throw new Error('E-Hentai: showkey not found')

    // Page 1 URL is available now from reader HTML
    const page1Url = (readerDoc.querySelector('#img') as HTMLImageElement | null)?.src ?? ''

    // Build pages + tokens. pages[0] has the real URL; rest are empty placeholders.
    const pages  = new Array<string>(pageCount).fill('')
    pages[0]     = page1Url
    const tokens = Array.from({ length: pageCount }, (_, i) =>
      encodeToken(gid, i + 1, galleryToken, showkey),
    )

    return { url: chapterUrl, pages, tokens }
  },

  /** Resolve one page URL lazily on viewport entry.
   *  Fetches the required thumb page on-demand (shared Promise, cached). */
  async resolvePageUrl(
    _manifest:   SourceManifest,
    token:       string,
    userCookies: Record<string, string>,
  ): Promise<string> {
    const d = decodeToken(token)
    if (!d) throw new Error(`E-Hentai: invalid token "${token}"`)
    return resolveUrl(d.gid, d.page, d.galleryToken, d.showkey, userCookies)
  },
}
