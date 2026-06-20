// HentaiFox adapter.
//
// Why adapter instead of declarative manifest:
//   - Page images use per-page file extensions stored in a JS variable
//     `g_th = $.parseJSON('{"1":"w,W,H","2":"j,W,H",...}')` where
//     the first char maps: j→jpg, p→png, w→webp, g→gif, b→bmp.
//   - The CDN host is chosen at runtime by JS (based on `unique_id`):
//     unique_id > 140236 → i3.hentaifox.com, else random i/i2.
//   - Both of these require imperative logic that the declarative
//     selector+template engine cannot express.
//
// What this adapter does:
//   1. Fetch /g/{galleryId}/1/ (reader page for page 1)
//   2. Parse hidden inputs: pages, image_dir, gallery_id, unique_id
//   3. Parse g_th JSON from inline <script> to get per-page ext
//   4. Select CDN host: unique_id > 140236 → i3, else i
//   5. Build full URL list: {cdn}/{dir}/{hash}/{n}.{ext}

import { fetchSource } from '../proxy'
import type { ChapterPages, SourceManifest } from '../manifest/types'
import type { SourceAdapter } from './types'

const EXT_MAP: Record<string, string> = {
  j: 'jpg',
  p: 'png',
  w: 'webp',
  g: 'gif',
  b: 'bmp',
}

function pickCdn(uniqueId: number): string {
  return uniqueId > 140236 ? 'i3.hentaifox.com' : 'i.hentaifox.com'
}

async function fetchReaderPage(
  galleryId:   string,
  userCookies: Record<string, string>,
): Promise<Document> {
  const url = `https://hentaifox.com/g/${galleryId}/1/`
  const headers: Record<string, string> = {
    Referer: 'https://hentaifox.com/',
  }
  if (Object.keys(userCookies).length > 0) {
    headers['Cookie'] = Object.entries(userCookies)
      .map(([k, v]) => `${k}=${v}`)
      .join('; ')
  }
  const res = await fetchSource(url, { headers })
  if (!res.ok) throw new Error(`HentaiFox reader HTTP ${res.status}`)
  const html = await res.text()
  return new DOMParser().parseFromString(html, 'text/html')
}

function parseGth(doc: Document): Record<string, string> {
  // g_th = $.parseJSON('{"1":"w,1280,1786",...}')
  for (const script of Array.from(doc.querySelectorAll('script'))) {
    const text = script.textContent ?? ''
    if (!text.includes('g_th')) continue
    const m = /g_th\s*=\s*(?:\$\.parseJSON|JSON\.parse)\s*\(\s*['"](\{.+?\})['"]\s*\)/.exec(text)
    if (m?.[1]) {
      try { return JSON.parse(m[1]) } catch { /* try next script */ }
    }
  }
  return {}
}

function inputVal(doc: Document, id: string): string {
  return (doc.querySelector(`input#${id}`) as HTMLInputElement | null)?.value ?? ''
}

export const hentaifoxAdapter: SourceAdapter = {
  async fetchChapterPages(
    _manifest:   SourceManifest,
    chapterUrl:  string,
    userCookies: Record<string, string>,
  ): Promise<ChapterPages> {
    // Extract galleryId from any HentaiFox URL shape:
    //   https://hentaifox.com/gallery/163079/
    //   https://hentaifox.com/g/163079/1/
    const match = /hentaifox\.com\/(?:gallery|g)\/(\d+)/.exec(chapterUrl)
    if (!match?.[1]) throw new Error(`Cannot extract galleryId from: ${chapterUrl}`)
    const galleryId = match[1]

    const doc       = await fetchReaderPage(galleryId, userCookies)
    const totalStr  = inputVal(doc, 'pages')
    const imageDir  = inputVal(doc, 'image_dir')
    const galleryHash = inputVal(doc, 'gallery_id')
    const uniqueId  = parseInt(inputVal(doc, 'unique_id') || '0', 10)
    const total     = parseInt(totalStr, 10)

    if (!total || !imageDir || !galleryHash) {
      throw new Error(`HentaiFox: missing reader metadata (pages=${totalStr} dir=${imageDir} hash=${galleryHash})`)
    }

    const gth  = parseGth(doc)
    const cdn  = pickCdn(uniqueId)
    const base = `https://${cdn}/${imageDir}/${galleryHash}`

    const pages: string[] = []
    for (let i = 1; i <= total; i++) {
      const raw = gth[String(i)] ?? ''
      const extCode = raw.split(',')[0] ?? 'j'
      const ext = EXT_MAP[extCode] ?? 'jpg'
      pages.push(`${base}/${i}.${ext}`)
    }

    return { url: chapterUrl, pages }
  },
}
