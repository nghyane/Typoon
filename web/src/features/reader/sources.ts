// Reader page source — common shape for saved archives and raw streams.
//
// Concrete implementations:
//
//   RawOnlineSource          source-adapter URLs (CORS proxied)
//   RawOfflineSource         Bunle.from(IDB blob, kind='raw')
//
// All expose the same surface so PagerView/StripView don't branch
// based on origin. URLs are lazy ObjectURLs (created on demand,
// revoked on close).

import { Bunle, type PageInfo } from '@nghyane/bunle'
import type { SourceFetch } from '@features/browse/proxy'

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

// ── Stable proxy — React-safe source swapping ───────────────────────

/**
 * Wraps a ReaderSource so React components always hold the same object
 * reference.  When the underlying source changes (cache eviction,
 * re-open), `swap()` replaces the delegate and defers closing the old
 * source until the next microtask — after React has committed the
 * re-render with the new source.
 */
export class StableSourceProxy implements ReaderSource {
  private _current: ReaderSource
  private _closed = false

  constructor(source: ReaderSource) {
    this._current = source
  }

  get mode(): ReaderMode { return this._current.mode }
  get pageCount(): number { return this._current.pageCount }
  get pages(): ReaderPage[] { return this._current.pages }

  getUrl(index: number): Promise<string> {
    if (this._closed) throw new DOMException('source closed', 'AbortError')
    return this._current.getUrl(index)
  }

  swap(next: ReaderSource): void {
    const prev = this._current
    this._current = next
    // Defer close — React may still be rendering with the old source
    // reference via usePageUrl's mountedRef check.
    if (prev && prev !== next) {
      queueMicrotask(() => prev.close())
    }
  }

  close(): void {
    this._closed = true
    this._current.close()
  }
}

export type ReaderMode =
  | 'raw-online'
  | 'raw-offline'


// ── Translated (BNL-backed) ──────────────────────────────────────────

class BunleReaderSource implements ReaderSource {
  readonly mode:      ReaderMode
  readonly pages:     ReaderPage[]
  readonly pageCount: number
  private readonly _bnl: Bunle
  private readonly _cache = new Map<number, string>()
  private readonly _pending = new Map<number, Promise<string>>()
  private _closed = false

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
    if (this._closed) throw new DOMException('source closed', 'AbortError')

    const cached = this._cache.get(index)
    if (cached) return Promise.resolve(cached)

    const pending = this._pending.get(index)
    if (pending) return pending

    const promise = this._bnl.url(index)
      .then(url => {
        if (this._closed) {
          URL.revokeObjectURL(url)
          throw new DOMException('source closed', 'AbortError')
        }
        this._cache.set(index, url)
        return url
      })
      .finally(() => this._pending.delete(index))

    this._pending.set(index, promise)
    return promise
  }

  close(): void {
    console.trace('[BunleReaderSource] close() called')
    this._closed = true
    this._pending.clear()
    this._cache.clear()
    this._bnl.close()
  }
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
  private readonly _sourceFetch: SourceFetch
  private readonly _cache = new Map<number, string>()
  private readonly _pending = new Map<number, Promise<string>>()
  private _closed = false

  constructor(rawUrls: string[], sourceFetch: SourceFetch) {
    this._urls         = rawUrls
    this._sourceFetch  = sourceFetch
    this.pageCount = rawUrls.length
    this.pages     = rawUrls.map((_, i) => ({ index: i, width: null, height: null }))
  }

  async getUrl(index: number): Promise<string> {
    if (this._closed) throw new DOMException('source closed', 'AbortError')

    const cached = this._cache.get(index)
    if (cached) return cached

    const pending = this._pending.get(index)
    if (pending) return pending

    const promise = this.fetchPageUrl(index)
      .finally(() => this._pending.delete(index))

    this._pending.set(index, promise)
    return promise
  }

  private async fetchPageUrl(index: number): Promise<string> {
    const raw = this._urls[index]
    if (!raw) throw new RangeError(`Page ${index} out of range`)

    // Adapter URLs go through the CDN proxy so the browser can load
    // them cross-origin and we can keep referers / cookies off the
    // upstream host.  CDN worker returns Access-Control-Allow-Origin: *
    // so fetch() from JS works without CORS issues.
    const proxied = this._sourceFetch.toBrowserUrl(raw)
    const res = await fetch(proxied)
    if (!res.ok) throw new Error(`Page ${index} fetch failed: ${res.status}`)
    const blob = await res.blob()
    if (this._closed) throw new DOMException('source closed', 'AbortError')
    const url  = URL.createObjectURL(blob)
    this._cache.set(index, url)
    return url
  }

  close(): void {
    console.warn('[RawOnlineSource] close()')
    this._closed = true
    for (const u of this._cache.values()) URL.revokeObjectURL(u)
    this._cache.clear()
    this._pending.clear()
  }
}

export function openRawOnline(rawUrls: string[], sourceFetch: SourceFetch): ReaderSource {
  return new RawOnlineSource(rawUrls, sourceFetch)
}
