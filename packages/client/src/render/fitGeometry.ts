import type { BBox, Polygon } from '../domain/geometry'
import type { TextPlacement } from '../domain/planning'

const DEFAULT_INSET = 2
const ELLIPSE_FIT_SCALE = 0.85
const DEFAULT_GROW_X = 1.28
const DEFAULT_GROW_Y = 1.22

/** How much to grow the source-text footprint to make room for the translation. */
export interface AnchorGrowth {
  readonly x: number
  readonly y: number
}

export interface FitRect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly rotationDeg: number
}

export function drawableRect(placement: TextPlacement): FitRect {
  // Honor an oriented (tilted) drawable for ANY role.  Tilted dialogue/narration
  // — e.g. chat bubbles on a phone screen — must render along its baseline, not
  // forced axis-aligned.  A single tilted block already carries a 4-point
  // oriented polygon from drawableForBlock; respect its angle.
  if (placement.drawable.length === 4 && polygonAngleDeg(placement.drawable) !== 0) {
    return orientedRect(placement.drawable)
  }
  // SFX grown from an axis-aligned bbox but still carrying a rotation.
  if (placement.role === 'sfx' && Math.abs(placement.rotationDeg) > 0.1) {
    return rotatedAabbRect(placement.bbox, placement.rotationDeg)
  }
  return axisAlignedRect(placement.drawable)
}

/**
 * T2 anchor region: centred on the OCR text footprint, grown by `growth` to make
 * room for the translation. The reliable signal is the source text geometry, not
 * the bubble — the bubble only acts as a soft ceiling later (fitLayout).
 */
export function textFitRect(placement: TextPlacement, growth: AnchorGrowth = { x: DEFAULT_GROW_X, y: DEFAULT_GROW_Y }): FitRect {
  const base = drawableRect(placement)
  if (placement.role === 'sfx' || Math.abs(base.rotationDeg) > 0.1 || placement.textBoxes.length === 0) return base

  // Grouped placements: use full drawable rect, text fills the semantic unit.
  if (placement.sourceUnitIds.length > 1) return base

  const textBox = unionTextBoxes(placement.textBoxes)
  const textCx = (textBox[0] + textBox[2]) / 2
  const textCy = (textBox[1] + textBox[3]) / 2
  const textW = Math.max(1, textBox[2] - textBox[0])
  const textH = Math.max(1, textBox[3] - textBox[1])
  const width = Math.min(base.width, Math.max(textW * growth.x, base.width * 0.58))
  const height = Math.min(base.height, Math.max(textH * growth.y, base.height * 0.58))

  return {
    x: clamp(textCx - width / 2, base.x, base.x + base.width - width),
    y: clamp(textCy - height / 2, base.y, base.y + base.height - height),
    width,
    height,
    rotationDeg: base.rotationDeg,
  }
}

export function rectBBox(rect: FitRect): BBox {
  return [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]
}

export function rectFromBBox(bbox: BBox, rotationDeg = 0): FitRect {
  return {
    x: bbox[0],
    y: bbox[1],
    width: Math.max(0, bbox[2] - bbox[0]),
    height: Math.max(0, bbox[3] - bbox[1]),
    rotationDeg,
  }
}

function polygonAngleDeg(polygon: Polygon): number {
  const p0 = polygon[0]
  const p1 = polygon[1]
  if (!p0 || !p1) return 0
  const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180 / Math.PI
  return Math.abs(angle) > 0.1 ? angle : 0
}

function rotatedAabbRect(bbox: BBox, rotationDeg: number): FitRect {
  const bboxW = Math.max(0, bbox[2] - bbox[0])
  const bboxH = Math.max(0, bbox[3] - bbox[1])
  const [rawW, rawH] = solveRotatedLocalSize(bboxW, bboxH, rotationDeg)
  const width = Math.max(0, rawW - DEFAULT_INSET * 2)
  const height = Math.max(0, rawH - DEFAULT_INSET * 2)
  const cx = (bbox[0] + bbox[2]) / 2
  const cy = (bbox[1] + bbox[3]) / 2
  return {
    x: cx - width / 2,
    y: cy - height / 2,
    width,
    height,
    rotationDeg,
  }
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

function orientedRect(polygon: Polygon): FitRect {
  const [p0, p1, p2] = polygon
  if (!p0 || !p1 || !p2) return axisAlignedRect(polygon)
  const widthRaw = distance(p0, p1)
  const heightRaw = distance(p1, p2)
  const width = Math.max(0, widthRaw - DEFAULT_INSET * 2)
  const height = Math.max(0, heightRaw - DEFAULT_INSET * 2)
  const center = polygonCenter(polygon)
  return {
    x: center[0] - width / 2,
    y: center[1] - height / 2,
    width,
    height,
    rotationDeg: Math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180 / Math.PI,
  }
}

function axisAlignedRect(polygon: Polygon): FitRect {
  const xs = polygon.map(p => p[0])
  const ys = polygon.map(p => p[1])
  const x1 = Math.min(...xs)
  const y1 = Math.min(...ys)
  const x2 = Math.max(...xs)
  const y2 = Math.max(...ys)
  const scale = polygon.length > 4 ? ELLIPSE_FIT_SCALE : 1
  const rawW = Math.max(0, x2 - x1 - DEFAULT_INSET * 2)
  const rawH = Math.max(0, y2 - y1 - DEFAULT_INSET * 2)
  const width = rawW * scale
  const height = rawH * scale
  return {
    x: x1 + DEFAULT_INSET + (rawW - width) / 2,
    y: y1 + DEFAULT_INSET + (rawH - height) / 2,
    width,
    height,
    rotationDeg: 0,
  }
}

function distance(a: readonly [number, number], b: readonly [number, number]): number {
  return Math.hypot(b[0] - a[0], b[1] - a[1])
}

function polygonCenter(polygon: Polygon): readonly [number, number] {
  const sum = polygon.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]] as [number, number], [0, 0])
  return [sum[0] / polygon.length, sum[1] / polygon.length]
}

function unionTextBoxes(boxes: readonly BBox[]): BBox {
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (const box of boxes) {
    x1 = Math.min(x1, box[0])
    y1 = Math.min(y1, box[1])
    x2 = Math.max(x2, box[2])
    y2 = Math.max(y2, box[3])
  }
  return [x1, y1, x2, y2]
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}
