// reader/pageProvider.ts — loads source page blobs, measures size, bounds memory.
//
// Gap fix: the old engine kept every loaded page blob for the chapter lifetime
// (in addition to the reader's own cache). Long chapters on mobile could OOM.
// This provider caps decoded pages and evicts least-recently-used entries that
// fall outside the active viewport.

import type { SourcePageSize } from '../pipeline/chapterContent'
import { throwIfAborted } from './asyncSignal'

export interface LoadedPage {
  readonly index: number
  readonly blob: Blob
  readonly size: SourcePageSize
}

export type ReadPageFn = (index: number, signal?: AbortSignal) => Promise<Blob>

export interface PageProviderOptions {
  readonly pageCount: number
  readonly maxCachedPages: number
  readonly readPage: ReadPageFn
  readonly onProgress?: (loadedPages: number) => void
  /**
   * Optional authoritative source size per page. When provided, it is the single
   * source of truth for page geometry: the provider skips its own decode so that
   * `unit.source` (the overlay's % denominator) is byte-identical to the size
   * the renderer uses for the page frame's aspect ratio. Falls back to decoding
   * when it returns null.
   */
  readonly pageSize?: (index: number) => SourcePageSize | null
}

export class PageProvider {
  private readonly cache = new Map<number, LoadedPage>()
  private readonly order: number[] = []
  private readonly sizes: Array<SourcePageSize | null>
  private loadedCount = 0

  constructor(private readonly options: PageProviderOptions) {
    this.sizes = new Array<SourcePageSize | null>(options.pageCount).fill(null)
  }

  size(index: number): SourcePageSize | null {
    return this.options.pageSize?.(index) ?? this.sizes[index] ?? this.cache.get(index)?.size ?? null
  }

  async read(index: number, signal: AbortSignal): Promise<LoadedPage> {
    const cached = this.cache.get(index)
    if (cached) {
      this.touch(index)
      return cached
    }
    throwIfAborted(signal)
    const blob = await this.options.readPage(index, signal)
    const size = this.options.pageSize?.(index) ?? await readImageSize(blob)
    const page: LoadedPage = { index, blob, size }
    this.cache.set(index, page)
    this.order.push(index)
    this.sizes[index] = size
    this.loadedCount += 1
    this.options.onProgress?.(this.loadedCount)
    return page
  }

  /** Evict LRU pages beyond the cap, keeping the given indexes resident. */
  evictExcept(keep: Iterable<number>): void {
    const protectedSet = new Set(keep)
    while (this.cache.size > this.options.maxCachedPages && this.order.length) {
      const candidate = this.order.find(index => !protectedSet.has(index))
      if (candidate === undefined) break
      this.order.splice(this.order.indexOf(candidate), 1)
      this.cache.delete(candidate)
    }
  }

  /** Preload image dimensions for all pages without keeping blobs in cache. */
  async preloadSizes(signal: AbortSignal): Promise<void> {
    const pending: number[] = []
    for (let i = 0; i < this.options.pageCount; i++) {
      if (this.options.pageSize?.(i)) continue
      if (this.sizes[i] === null) pending.push(i)
    }
    if (!pending.length) return

    const concurrency = 6
    for (let batch = 0; batch < pending.length; batch += concurrency) {
      throwIfAborted(signal)
      const slice = pending.slice(batch, batch + concurrency)
      await Promise.all(slice.map(async i => {
        const blob = await this.options.readPage(i, signal)
        const size = await readImageSize(blob)
        this.sizes[i] = size
      }))
    }
  }

  clear(): void {
    this.cache.clear()
    this.order.length = 0
    this.sizes.fill(null)
    this.loadedCount = 0
  }

  private touch(index: number): void {
    const at = this.order.indexOf(index)
    if (at >= 0) this.order.splice(at, 1)
    this.order.push(index)
  }
}

async function readImageSize(blob: Blob): Promise<SourcePageSize> {
  // `from-image`: honor EXIF orientation so decoded W/H matches what <img>
  // displays and the OCR canvas captures — keeping every coordinate space
  // (display, OCR, overlay) in one orientation.
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' })
  try {
    return { width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}
