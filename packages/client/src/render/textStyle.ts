import type { TextPlacement } from '../domain/planning'

const DEFAULT_FONT_SIZE_PX = 24
const DIALOGUE_STROKE_RATIO = 0.08
const SFX_STROKE_RATIO = 0.10

export interface TextStrokePlan {
  readonly color: string
  readonly widthPx: number
}

export interface TextStylePlan {
  readonly fill: string
  readonly fontWeight: string
  readonly strokes: readonly TextStrokePlan[]
  readonly shadow: string | null
  readonly backgroundColor: string | null
}

export function buildTextStyle(placement: TextPlacement, fontSizePx = DEFAULT_FONT_SIZE_PX, backgroundRgb?: readonly [number, number, number] | null): TextStylePlan {
  if (placement.role !== 'sfx') {
    return {
      fill: '#111111',
      fontWeight: '700',
      strokes: [{ color: '#ffffff', widthPx: strokeWidth(fontSizePx, DIALOGUE_STROKE_RATIO, 2, 18) }],
      shadow: null,
      backgroundColor: placement.role === 'narration' && backgroundRgb
        ? `rgb(${backgroundRgb[0]},${backgroundRgb[1]},${backgroundRgb[2]})`
        : null,
    }
  }

  return {
    fill: '#111111',
    fontWeight: '800',
    strokes: [{ color: '#ffffff', widthPx: strokeWidth(fontSizePx, SFX_STROKE_RATIO, 4, 28) }],
    shadow: null,
    backgroundColor: null,
  }
}

function strokeWidth(fontSizePx: number, ratio: number, minPx: number, maxPx: number): number {
  return Math.min(maxPx, Math.max(minPx, Math.round(fontSizePx * ratio)))
}
