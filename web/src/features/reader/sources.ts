// Reader page source — common shape for translated archives and raw streams.
//
// Four concrete implementations:
//
//   TranslatedOnlineSource   Bunle.open(presigned archive URL)
//   TranslatedOfflineSource  Bunle.from(IndexedDB blob)
//   RawOnlineSource          source-adapter URLs (CORS proxied)
//   RawOfflineSource         Bunle.from(IDB blob, kind='raw')
//
// All expose the same surface so PagerView/StripView don't branch
// based on origin. URLs are lazy ObjectURLs (created on demand,
// revoked on close).

import { Bunle, type PageInfo } from '@nghyane/bunle'
import { proxify } from '@features/browse/proxy'

export interface ReaderPage {
  index:  number
  width:  number | null         // null when unknown ahead of time (raw streaming)
  height: number | null
}

export interface ReaderSource {
  readonly mode:      ReaderMode
  readonly pageCount: number
  readonly pages:     ReaderPage[]
  /** Resolve an ObjectURL for a page; safe to call multiple times. */
  getUrl(index: number): Promise<string>
  /** Free all cached ObjectURLs. Call from `useEffect` cleanup. */
  close(): void
}

export type ReaderMode =
  | 'translated-online'
  | 'translated-offline'
  | 'raw-online'
  | 'raw-offline'


// ── Translated (BNL-backed) ──────────────────────────────────────────

class BunleReaderSource implements ReaderSource {
  readonly mode:      ReaderMode
  readonly pages:     ReaderPage[]
  readonly pageCount: number
  private readonly _bnl: Bunle

  constructor(bnl: Bunle, mode: ReaderMode) {
    this._bnl    = bnl
    this.mode    = mode
    this.pages   = bnl.pages.map((p: PageInfo) => ({
      index:  p.index,
      width:  p.width,
      height: p.height,
    }))
    this.pageCount = bnl.pageCount
  }

  getUrl(index: number): Promise<string> {
    return this._bnl.url(index)
  }

  close(): void {
    this._bnl.close()
  }
}

export async function openTranslatedOnline(archiveUrl: string): Promise<ReaderSource> {
  const bnl = await Bunle.open(archiveUrl)
  return new BunleReaderSource(bnl, 'translated-online')
}

export function openTranslatedOffline(blob: Blob): ReaderSource | Promise<ReaderSource> {
  return blob.arrayBuffer().then(buf => new BunleReaderSource(Bunle.from(buf), 'translated-offline'))
}

export function openRawOffline(blob: Blob): Promise<ReaderSource> {
  return blob.arrayBuffer().then(buf => new BunleReaderSource(Bunle.from(buf), 'raw-offline'))
}


// ── Raw online (adapter stream) ──────────────────────────────────────

class RawOnlineSource implements ReaderSource {
  readonly mode:      ReaderMode = 'raw-online'
  readonly pages:     ReaderPage[]
  readonly pageCount: number
  private readonly _urls: string[]
  private readonly _cache = new Map<number, string>()

  constructor(rawUrls: string[]) {
    this._urls     = rawUrls
    this.pageCount = rawUrls.length
    this.pages     = rawUrls.map((_, i) => ({ index: i, width: null, height: null }))
  }

  async getUrl(index: number): Promise<string> {
    const cached = this._cache.get(index)
    if (cached) return cached

    const raw = this._urls[index]
    if (!raw) throw new RangeError(`Page ${index} out of range`)

    // Adapter URLs go through the CDN proxy so the browser can load
    // them cross-origin and we can keep referers / cookies off the
    // upstream host.
    const proxied = proxify(raw)
    const res = await fetch(proxied)
    if (!res.ok) throw new Error(`Page ${index} fetch failed: ${res.status}`)
    const blob = await res.blob()
    const url  = URL.createObjectURL(blob)
    this._cache.set(index, url)
    return url
  }

  close(): void {
    for (const u of this._cache.values()) URL.revokeObjectURL(u)
    this._cache.clear()
  }
}

export function openRawOnline(rawUrls: string[]): ReaderSource {
  return new RawOnlineSource(rawUrls)
}
