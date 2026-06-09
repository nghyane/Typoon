import type { BBox, Polygon } from './geometry'
import type { TextRole } from './planning'

export type TextDirection = 'horizontal' | 'vertical'

export interface TextWord {
  readonly bbox: BBox
  readonly text: string
  readonly textSeparator?: string
}

export interface TextLine {
  readonly bbox: BBox
  readonly text: string
  readonly rotationDeg: number
  readonly words: readonly TextWord[]
}

export interface TextBlock {
  readonly bbox: BBox
  readonly polygon: Polygon
  readonly text: string
  readonly rotationDeg: number
  readonly textDirection: TextDirection
  readonly confidence: number
  readonly lines: readonly TextLine[]
  readonly words: readonly TextWord[]
}

export interface TextUnit {
  readonly id: string
  readonly pageIndex: number
  readonly blockIds: readonly string[]
  readonly sourceText: string
  readonly roleHint?: TextRole
}

export interface RecognizedTextPage {
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly detectedLanguage: string | null
  readonly blocks: readonly TextBlock[]
  readonly timingMs: Record<string, number>
}
