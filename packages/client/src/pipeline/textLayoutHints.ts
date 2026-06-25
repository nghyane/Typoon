import type { TextDirection } from '../domain/text'
import type { TextLayoutHint, TextLayoutKind, TextPlacement } from '../domain/planning'

const VERTICAL_ASPECT = 5
const SOURCE_VERTICAL_ASPECT = 1.6
const VERTICAL_MAX_WIDTH_RATIO = 0.14

export function withTextLayoutHints(placements: readonly TextPlacement[], pageSize: readonly [number, number]): TextPlacement[] {
  return placements.map(placement => ({
    ...placement,
    layoutHint: inferLayoutHint(placement, pageSize),
  }))
}

function inferLayoutHint(placement: TextPlacement, pageSize: readonly [number, number]): TextLayoutHint {
  const direction = textDirection(placement, pageSize)

  if (direction === 'vertical') {
    return {
      direction: 'vertical',
      inlineAlign: 'center',
      blockAlign: 'center',
      kind: 'vertical-label',
      confidence: 0.85,
      reason: 'source-vertical',
    }
  }

  return {
    direction: 'horizontal',
    inlineAlign: 'center',
    blockAlign: 'center',
    kind: layoutKind(placement, pageSize[0]),
    confidence: 0.65,
    reason: 'center-default',
  }
}

function textDirection(placement: TextPlacement, pageSize: readonly [number, number]): TextDirection {
  const [pageW] = pageSize
  const bw = bboxW(placement.bbox)
  const bh = bboxH(placement.bbox)
  if (placement.fontHint?.sourceDirection === 'vertical'
    && bw / Math.max(1, pageW) <= VERTICAL_MAX_WIDTH_RATIO
    && (bh / Math.max(1, bw) >= SOURCE_VERTICAL_ASPECT || majorityTallTextBoxes(placement))) return 'vertical'
  if (bh / Math.max(1, bw) >= VERTICAL_ASPECT && bw / Math.max(1, pageW) <= VERTICAL_MAX_WIDTH_RATIO) return 'vertical'
  return 'horizontal'
}

function majorityTallTextBoxes(placement: TextPlacement): boolean {
  if (!placement.textBoxes.length) return false
  const tall = placement.textBoxes.filter(box => bboxH(box) / Math.max(1, bboxW(box)) >= 2).length
  return tall * 2 >= placement.textBoxes.length
}

function layoutKind(placement: TextPlacement, pageWidth: number): TextLayoutKind {
  if (placement.role === 'sfx') return 'decorative'
  // Only treat as banner (single-line) if the bbox is wide AND short enough
  // to strongly suggest a title strip rather than a dialogue bubble.
  // A wide but not very tall box with only 1 detected text line and an
  // aspect ratio > 5:1 suggests a banner.
  const bw = bboxW(placement.bbox)
  const bh = bboxH(placement.bbox)
  const widthRatio = bw / Math.max(1, pageWidth)
  const aspectOk = bh > 0 && bw / bh >= 5
  if (placement.textBoxes.length <= 1 && widthRatio >= 0.20 && aspectOk) return 'banner'
  return 'paragraph'
}

function bboxW(bbox: readonly number[]): number {
  return Math.max(0, bbox[2] - bbox[0])
}

function bboxH(bbox: readonly number[]): number {
  return Math.max(0, bbox[3] - bbox[1])
}

export function defaultLayoutHint(): TextLayoutHint {
  return {
    direction: 'horizontal',
    inlineAlign: 'center',
    blockAlign: 'center',
    kind: 'paragraph',
    confidence: 0,
    reason: 'pending',
  }
}
