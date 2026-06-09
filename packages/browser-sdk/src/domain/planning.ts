import type { BBox, Polygon } from './geometry'
import type { TextDirection } from './text'
import type { TranslationUnit } from './translation'

export type TextRole = 'dialogue' | 'sfx' | 'narration'

export interface FontHint {
  readonly sourceFontPx?: number
  readonly sourceLineCount?: number
  readonly sourceAvgCharsPerLine?: number
  readonly sourceDirection: TextDirection
}

export interface TextPlacement {
  readonly id: string
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly sourceText: string
  readonly drawable: Polygon
  readonly bbox: BBox
  readonly textBoxes: readonly BBox[]
  readonly role: TextRole
  readonly rotationDeg: number
  readonly confidence: number
  readonly fontHint: FontHint | null
}

export interface PagePlan {
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly placements: readonly TextPlacement[]
  readonly units: readonly TranslationUnit[]
  readonly timingMs: Record<string, number>
}
