import type { BBox } from './geometry'

export type TextRegionKind = 'bubble' | 'text_bubble' | 'text_free'

export interface TextRegion {
  readonly kind: TextRegionKind
  readonly bbox: BBox
  readonly confidence: number
}

export interface PageTextRegions {
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly regions: readonly TextRegion[]
  readonly timingMs: Record<string, number>
}
