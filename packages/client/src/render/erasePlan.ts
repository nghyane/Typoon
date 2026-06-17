import type { BBox } from '../domain/geometry'
import type { TextPlacement } from '../domain/planning'
import { hasReliableBackgroundFill, type SafeMarginsDebug } from './backgroundFit'

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
    if (!hasReliableBackgroundFill(margin)) continue
    const fill = `rgb(${margin.backgroundRgb[0]},${margin.backgroundRgb[1]},${margin.backgroundRgb[2]})`

    for (const bbox of textEraseBoxes(placement.textBoxes, placement.bbox)) {
      // Non-SFX render is axis-aligned, erase must match.
      shapes.push(rotatedRectShape(bbox, 0, fill))
    }
  }
  return { kind: 'flat-fill', shapes }
}

function rotatedRectShape(bbox: BBox, rotationDeg: number, fill: string): EraseShape {
  const [x1, y1, x2, y2] = bbox
  const cx = (x1 + x2) / 2
  const cy = (y1 + y2) / 2
  const bboxW = Math.max(1, x2 - x1)
  const bboxH = Math.max(1, y2 - y1)
  const [width, height] = Math.abs(rotationDeg) > 0.1
    ? solveRotatedLocalSize(bboxW, bboxH, rotationDeg)
    : [bboxW, bboxH]
  return { kind: 'rotated-rect', cx, cy, width, height, rotationDeg, radius: 2, fill }
}

function textEraseBoxes(lineBoxes: readonly BBox[], fallback: BBox): BBox[] {
  const boxes = lineBoxes.length ? lineBoxes : [fallback]
  return boxes.map(box => {
    const w = box[2] - box[0]
    const h = box[3] - box[1]
    const pad = Math.max(3, Math.min(12, Math.round(Math.min(w, h) * 0.18)))
    return [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad] as BBox
  })
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
