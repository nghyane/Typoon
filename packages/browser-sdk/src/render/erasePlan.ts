import type { BBox } from '../domain/geometry'
import type { ImagePixels } from '../domain/image'
import type { TextPlacement } from '../domain/planning'

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
}

export function buildErasePlan(
  placements: readonly TextPlacement[],
  image: ImagePixels | undefined,
  options: BuildErasePlanOptions = {},
): ErasePlan {
  const strategy = options.strategy ?? 'flat-fill'
  if (strategy === 'none') return { kind: 'none' }

  const shapes: EraseShape[] = []
  for (const placement of placements) {
    if (placement.role === 'sfx') continue
    for (const bbox of textEraseBoxes(placement.textBoxes, placement.bbox)) {
      shapes.push(rotatedRectShape(bbox, placement.rotationDeg, sampleBackground(image, bbox)))
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

function sampleBackground(image: ImagePixels | undefined, bbox: BBox): string {
  if (!image) return 'rgb(255,255,255)'
  const samples: number[][] = []
  const [x1, y1, x2, y2] = bbox.map(Math.round) as unknown as BBox
  const ring = 5
  for (let y = y1 - ring; y <= y2 + ring; y += 2) {
    for (let x = x1 - ring; x <= x2 + ring; x += 2) {
      const inside = x1 <= x && x <= x2 && y1 <= y && y <= y2
      if (inside) continue
      const rgb = pixelAt(image, x, y)
      if (rgb) samples.push(rgb)
    }
  }
  if (!samples.length) return 'rgb(255,255,255)'
  samples.sort((a, b) => luminance(a) - luminance(b))
  const pick = samples[Math.floor(samples.length * 0.65)] ?? samples[samples.length - 1]!
  return `rgb(${pick[0]},${pick[1]},${pick[2]})`
}

function pixelAt(image: ImagePixels, x: number, y: number): number[] | null {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return null
  const i = (y * image.width + x) * 4
  return [image.data[i] ?? 255, image.data[i + 1] ?? 255, image.data[i + 2] ?? 255]
}

function luminance(rgb: readonly number[]): number {
  return (rgb[0] ?? 0) * 0.2126 + (rgb[1] ?? 0) * 0.7152 + (rgb[2] ?? 0) * 0.0722
}
