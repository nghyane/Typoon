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
    return this.sizes[index] ?? this.cache.get(index)?.size ?? null
  }

  async read(index: number, signal: AbortSignal): Promise<LoadedPage> {
    const cached = this.cache.get(index)
    if (cached) {
      this.touch(index)
      return cached
    }
    throwIfAborted(signal)
    const blob = await this.options.readPage(index, signal)
    const size = await readImageSize(blob)
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
  const bitmap = await createImageBitmap(blob)
  try {
    return { width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}
