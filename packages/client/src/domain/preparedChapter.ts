import type { ImagePixels } from './image'
import type { PageSize, Rect } from './source'

export interface PreparedChapter {
  readonly runId: string
  readonly pages: readonly PreparedPage[]
  readonly sourcePageToPreparedPage: readonly SourcePageMapping[]
}

export interface PreparedPage {
  readonly id: string
  readonly index: number
  readonly size: PageSize
  readonly asset: PreparedPageAsset
  readonly projections: readonly PageProjection[]
}

export interface PreparedPageAsset {
  readPixels(signal?: AbortSignal): Promise<ImagePixels>
  readBlob(signal?: AbortSignal): Promise<Blob>
  release(): void
}

export interface PageProjection {
  readonly sourcePageIndex: number
  readonly sourceRect: Rect
  readonly sourcePageSize?: PageSize
  readonly preparedRect: Rect
}

export interface SourcePageMapping {
  readonly sourcePageIndex: number
  readonly preparedPageIndex: number
}
