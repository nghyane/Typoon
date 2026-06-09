import type { ImagePixels } from '../domain/image'
import type { TextPlacement } from '../domain/planning'

export interface TextStrokePlan {
  readonly color: string
  readonly widthPx: number
}

export interface TextStylePlan {
  readonly fill: string
  readonly fontWeight: string
  readonly strokes: readonly TextStrokePlan[]
  readonly shadow: string | null
}

export function buildTextStyle(placement: TextPlacement, _image?: ImagePixels): TextStylePlan {
  if (placement.role !== 'sfx') {
    return { fill: '#111111', fontWeight: '700', strokes: [], shadow: null }
  }

  return {
    fill: '#111111',
    fontWeight: '800',
    strokes: [{ color: '#ffffff', widthPx: 4 }],
    shadow: null,
  }
}
