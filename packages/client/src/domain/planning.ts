import type { BBox, Polygon } from './geometry'
import type { TextDirection } from './text'

export type TextRole = 'dialogue' | 'sfx' | 'narration'
export type TextInlineAlign = 'left' | 'center'
export type TextBlockAlign = 'left' | 'center'
export type TextLayoutKind = 'paragraph' | 'banner' | 'vertical-label' | 'decorative'

export interface FontHint {
  readonly sourceFontPx?: number
  readonly sourceLineCount?: number
  readonly sourceAvgCharsPerLine?: number
  readonly sourceDirection: TextDirection
}

export interface TextLayoutHint {
  readonly direction: TextDirection
  readonly inlineAlign: TextInlineAlign
  readonly blockAlign: TextBlockAlign
  readonly kind: TextLayoutKind
  readonly confidence: number
  readonly reason: string
}

export interface TextPlacement {
  readonly id: string
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly sourceUnitIds: readonly string[]
  readonly drawable: Polygon
  readonly bbox: BBox
  readonly textBoxes: readonly BBox[]   // line-level (for fit, expansion)
  readonly wordBoxes: readonly BBox[]   // word-level (for tight erase)
  // Detected container (bubble/region) extent that bounds expansion, so a
  // white-on-white bubble whose background bleeds to the whole page does not let
  // the text grow past the real bubble. null when no container is known.
  readonly containerBBox: BBox | null
  readonly role: TextRole
  readonly rotationDeg: number
  readonly confidence: number
  readonly fontHint: FontHint | null
  readonly layoutHint: TextLayoutHint
}
