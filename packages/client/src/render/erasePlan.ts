import type { BBox } from '../domain/geometry'
import type { TextPlacement } from '../domain/planning'
import { hasReliableBackgroundFill, hasAnyBackgroundFill, type SafeMarginsDebug } from './backgroundFit'

export type EraseStrategy = 'none' | 'flat-fill'

export type ErasePlan =
  | { readonly kind: 'none' }
  | { readonly kind: 'flat-fill'; readonly shapes: readonly EraseShape[] }

export interface EraseShape {
  readonly kind: 'rotated-rect'
  readonly cx: number
  readonly cy: number
  readonly width: number
  readonly height: number
  readonly rotationDeg: number
  readonly radius: number
  readonly fill: string
}

export interface BuildErasePlanOptions {
  readonly strategy?: EraseStrategy
  readonly placementMargins?: readonly SafeMarginsDebug[]
}

export function buildErasePlan(
  placements: readonly TextPlacement[],
  options: BuildErasePlanOptions = {},
): ErasePlan {
  const strategy = options.strategy ?? 'flat-fill'
  if (strategy === 'none') return { kind: 'none' }

  const shapes: EraseShape[] = []
  for (let i = 0; i < placements.length; i++) {
    const placement = placements[i]!
    if (placement.role === 'sfx') continue

    const margin = options.placementMargins?.[i]
    if (!hasAnyBackgroundFill(margin)) continue
    // dialogue needs reliable component detection (bubble boundary); narration sits on flat bg
    if (placement.role === 'dialogue' && !hasReliableBackgroundFill(margin)) continue
    const fill = `rgb(${margin.backgroundRgb[0]},${margin.backgroundRgb[1]},${margin.backgroundRgb[2]})`
    const rot = Math.abs(placement.rotationDeg) > 1 ? placement.rotationDeg : 0

    if (rot) {
      // Per-word rotated rects — mirror the non-rotated path's per-word
      // strategy so each erase hugs a single word instead of inflating to a
      // union that captures inter-word gaps.  solveRotatedLocalSize is safe
      // here because a single word's axis-aligned bbox comes from ONE
      // oriented glyph (it breaks only on multi-word unions).
      for (const box of placement.wordBoxes) {
        const w = box[2] - box[0]
        const h = box[3] - box[1]
        const pad = 1
        const paddedW = w + pad * 2
        const paddedH = h + pad * 2
        const [ow, oh] = solveRotatedLocalSize(paddedW, paddedH, rot)
        shapes.push({
          kind: 'rotated-rect',
          cx: (box[0] + box[2]) / 2,
          cy: (box[1] + box[3]) / 2,
          width: ow,
          height: oh,
          rotationDeg: rot,
          radius: 2,
          fill,
        })
      }
    } else {
      for (const bbox of textEraseBoxes(placement.wordBoxes, placement.bbox)) {
        const clamped = intersectBBox(bbox, margin.safeBounds)
        if (!clamped) continue
        shapes.push(rotatedRectShape(clamped, fill))
      }
    }
  }
  return { kind: 'flat-fill', shapes }
}

function rotatedRectShape(bbox: BBox, fill: string): EraseShape {
  const [x1, y1, x2, y2] = bbox
  return {
    kind: 'rotated-rect',
    cx: (x1 + x2) / 2,
    cy: (y1 + y2) / 2,
    width: Math.max(1, x2 - x1),
    height: Math.max(1, y2 - y1),
    rotationDeg: 0,
    radius: 2,
    fill,
  }
}

function intersectBBox(box: BBox, bounds: BBox): BBox | null {
  const x1 = Math.max(box[0], bounds[0])
  const y1 = Math.max(box[1], bounds[1])
  const x2 = Math.min(box[2], bounds[2])
  const y2 = Math.min(box[3], bounds[3])
  if (x2 - x1 < 1 || y2 - y1 < 1) return null
  return [x1, y1, x2, y2]
}

function solveRotatedLocalSize(bboxW: number, bboxH: number, rotationDeg: number): readonly [number, number] {
  const rad = Math.abs(rotationDeg) * Math.PI / 180
  const c = Math.abs(Math.cos(rad))
  const s = Math.abs(Math.sin(rad))
  const det = c * c - s * s
  if (Math.abs(det) < 0.2) return [bboxW, bboxH]
  const width = (c * bboxW - s * bboxH) / det
  const height = (-s * bboxW + c * bboxH) / det
  if (width <= 0 || height <= 0 || !Number.isFinite(width) || !Number.isFinite(height)) {
    return [bboxW, bboxH]
  }
  return [width, height]
}

function textEraseBoxes(lineBoxes: readonly BBox[], fallback: BBox): BBox[] {
  // Per-line boxes hug the glyphs; the placement bbox is the drawable union
  // (with padding). Including it back fills inter-line gaps and corners,
  // bleeding past the actual text and potentially out of the bubble.
  const boxes = lineBoxes.length ? lineBoxes : [fallback]
  return boxes.map(box => {
    const w = box[2] - box[0]
    const h = box[3] - box[1]
    const pad = Math.max(3, Math.min(12, Math.round(Math.min(w, h) * 0.18)))
    return [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad] as BBox
  })
}
