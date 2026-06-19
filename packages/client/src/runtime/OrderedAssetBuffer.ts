import type { PageAsset } from '../domain/source'

type OrderedAsset = Pick<PageAsset, 'index'>

/**
 * Buffers loaded assets and emits them to the preparation stream
 * in the expected preparation order.
 *
 * Assets may load out of order (priority queue, network variance),
 * but preparation may require a different order. Identity preparation can use
 * priority order, while continuous-strip preparation uses natural source order.
 */
export class OrderedAssetBuffer<T extends OrderedAsset = PageAsset> {
  /** How many assets have been emitted so far. */
  private emitted = 0
  private readonly buffer = new Map<number, T>()
  private readonly skipped = new Set<number>()
  private readonly prepareOrder: readonly number[]

  constructor(prepareOrder: readonly number[]) {
    this.prepareOrder = prepareOrder
  }

  /**
   * Push a loaded asset.  Returns zero or more assets ready for
   * preparation, in source-index order.
   */
  push(index: number, asset: T): T[] {
    this.buffer.set(index, asset)
    return this.drainReady()
  }

  /** Mark a failed page as non-blocking and emit any later ready assets. */
  skip(index: number): T[] {
    this.skipped.add(index)
    return this.drainReady()
  }

  private drainReady(): T[] {
    const ready: T[] = []

    while (this.emitted < this.prepareOrder.length) {
      const nextIndex = this.prepareOrder[this.emitted]!
      if (this.skipped.has(nextIndex)) {
        this.emitted += 1
        continue
      }
      const item = this.buffer.get(nextIndex)
      if (!item) break
      ready.push(item)
      this.emitted += 1
    }

    return ready
  }
}
