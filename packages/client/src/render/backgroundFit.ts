import type { BBox } from '../domain/geometry'
import type { ImagePixels } from '../domain/image'
import type { TextPlacement } from '../domain/planning'
import type { FitRect } from './fitGeometry'
import { rectBBox } from './fitGeometry'

export type Rgb = readonly [number, number, number]
type Direction = 'left' | 'right' | 'top' | 'bottom'

const EXPAND_STEP_PX = 3
const MAX_EXPAND_STEPS = 200
const MIN_BACKGROUND_COVERAGE = 0.82
const COMPONENT_MAX_AREA_RATIO = 3.0
const COMPONENT_MAX_TALL_AREA_RATIO = 8.0
const COMPONENT_MAX_RADIUS_FRACTION = 0.08

export interface SafeMargins {
  readonly top: number
  readonly bottom: number
  readonly left: number
  readonly right: number
}

export interface SafeMarginsDebug {
  readonly reasons: {
    readonly top: string
    readonly bottom: string
    readonly left: string
    readonly right: string
    readonly overall: string
  }
  readonly margins: SafeMargins
  readonly backgroundRgb: Rgb | null
  readonly backgroundTolerance: number
  readonly safeBounds: BBox
  readonly componentBBox: BBox | null
  readonly componentConfidence: number
}

export function hasReliableBackgroundFill(margin: SafeMarginsDebug | null | undefined): margin is SafeMarginsDebug & { readonly backgroundRgb: Rgb } {
  return !!margin?.backgroundRgb && margin.componentConfidence >= 0.6
}

export function estimateSafeMargins(args: {
  readonly image: ImagePixels
  readonly placement: TextPlacement
  readonly baseRect: FitRect
  readonly obstacles: readonly BBox[]
  readonly pageSize: readonly [number, number]
}): SafeMarginsDebug {
  const base = roundedRectBBox(args.baseRect, args.pageSize)

  if (Math.abs(args.baseRect.rotationDeg) > 0.1) {
    return emptyDebug(base, 'rotated')
  }
  // Non-SFX placements are rendered axis-aligned; OCR rotation noise must not block margins.
  if (args.placement.role === 'sfx' && Math.abs(args.placement.rotationDeg) > 1) {
    return emptyDebug(base, 'rotated')
  }

  const sampleBoxes = args.placement.textBoxes.length ? args.placement.textBoxes : [base]
  const background = estimateBackground(args.image, sampleBoxes)
  if (!background) return emptyDebug(base, 'no-background')

  const selfBoxes = args.placement.textBoxes.length ? args.placement.textBoxes : [base]
  const component = detectBackgroundComponent({
    image: args.image,
    seed: base,
    selfBoxes,
    background,
    obstacles: args.obstacles,
    pageSize: args.pageSize,
  })

  const margins = marginsFromComponent(base, component.bbox)
  const reasons = marginsReasons(margins, component)

  return {
    reasons,
    margins,
    safeBounds: component.bbox,
    backgroundRgb: background.rgb,
    backgroundTolerance: background.tolerance,
    componentBBox: component.bbox,
    componentConfidence: component.confidence,
  }
}

function emptyDebug(base: BBox, reason: string): SafeMarginsDebug {
  return {
    reasons: { top: reason, bottom: reason, left: reason, right: reason, overall: reason },
    margins: { top: 0, bottom: 0, left: 0, right: 0 },
    safeBounds: base,
    backgroundRgb: null,
    backgroundTolerance: 0,
    componentBBox: null,
    componentConfidence: 0,
  }
}

function marginsFromComponent(base: BBox, component: BBox): SafeMargins {
  return {
    top: Math.max(0, base[1] - component[1]),
    bottom: Math.max(0, component[3] - base[3]),
    left: Math.max(0, base[0] - component[0]),
    right: Math.max(0, component[2] - base[2]),
  }
}

function marginsReasons(
  margins: SafeMargins,
  component: ComponentResult,
): SafeMarginsDebug['reasons'] {
  const dirReason = (px: number): string => px > 0 ? 'ok' : 'component-edge'
  const overall = (margins.top + margins.bottom + margins.left + margins.right) > 0
    ? (component.confidence >= 0.6 ? 'ok' : 'low-confidence')
    : 'blocked'
  return {
    top: dirReason(margins.top),
    bottom: dirReason(margins.bottom),
    left: dirReason(margins.left),
    right: dirReason(margins.right),
    overall,
  }
}

interface ComponentResult {
  readonly bbox: BBox
  readonly confidence: number
}

function detectBackgroundComponent(args: {
  readonly image: ImagePixels
  readonly seed: BBox
  readonly selfBoxes: readonly BBox[]
  readonly background: BackgroundModel
  readonly obstacles: readonly BBox[]
  readonly pageSize: readonly [number, number]
}): ComponentResult {
  const maxRadius = Math.max(
    args.pageSize[0] * COMPONENT_MAX_RADIUS_FRACTION,
    args.pageSize[1] * COMPONENT_MAX_RADIUS_FRACTION,
  )
  const seedArea = bboxArea(args.seed)
  const maxArea = seedArea * componentMaxAreaRatio(args.seed)

  let bounds: BBox = [...args.seed] as unknown as BBox

  for (let step = 0; step < MAX_EXPAND_STEPS; step++) {
    let changed = false
    for (const direction of (['top', 'bottom', 'left', 'right'] as const)) {
      if (bboxArea(bounds) >= maxArea) break
      if (exceedsRadius(bounds, args.seed, maxRadius)) break

      const next = grow(bounds, args.pageSize, direction)
      if (sameBBox(next, bounds)) continue

      const strip = growthStrip(bounds, next, direction)
      if (bboxWidth(strip) < 1 || bboxHeight(strip) < 1) continue

      if (args.obstacles.some(obstacle => intersects(strip, obstacle))) continue

      const isSelfText = args.selfBoxes.some(box => intersects(strip, box))
      if (isSelfText) {
        bounds = next
        changed = true
        continue
      }

      const stats = backgroundStats(args.image, strip, args.background)
      if (stats.samples === 0 || stats.coverage < MIN_BACKGROUND_COVERAGE) continue

      bounds = next
      changed = true
    }
    if (!changed) break
  }

  const componentArea = bboxArea(bounds)
  const coverageStats = backgroundStats(args.image, bounds, args.background)
  const areaRatio = componentArea / Math.max(1, seedArea)
  const confidence = computeComponentConfidence(coverageStats.coverage, areaRatio)

  return { bbox: bounds, confidence }
}

function componentMaxAreaRatio(seed: BBox): number {
  const width = Math.max(1, bboxWidth(seed))
  const height = Math.max(1, bboxHeight(seed))
  const aspect = Math.max(width, height) / Math.min(width, height)
  if (aspect >= 2.0) return COMPONENT_MAX_TALL_AREA_RATIO
  return COMPONENT_MAX_AREA_RATIO
}

function computeComponentConfidence(bgCoverage: number, areaRatio: number): number {
  let conf = bgCoverage
  if (areaRatio > 4) conf *= 0.85
  if (areaRatio > 5) conf *= 0.80
  return clamp(conf, 0, 1)
}

function exceedsRadius(bounds: BBox, seed: BBox, maxRadius: number): boolean {
  return (
    seed[0] - bounds[0] > maxRadius ||
    bounds[2] - seed[2] > maxRadius ||
    seed[1] - bounds[1] > maxRadius ||
    bounds[3] - seed[3] > maxRadius
  )
}

function grow(bounds: BBox, pageSize: readonly [number, number], direction: Direction): BBox {
  switch (direction) {
    case 'left': return [Math.max(0, bounds[0] - EXPAND_STEP_PX), bounds[1], bounds[2], bounds[3]]
    case 'right': return [bounds[0], bounds[1], Math.min(pageSize[0], bounds[2] + EXPAND_STEP_PX), bounds[3]]
    case 'top': return [bounds[0], Math.max(0, bounds[1] - EXPAND_STEP_PX), bounds[2], bounds[3]]
    case 'bottom': return [bounds[0], bounds[1], bounds[2], Math.min(pageSize[1], bounds[3] + EXPAND_STEP_PX)]
  }
}

function growthStrip(before: BBox, after: BBox, direction: Direction): BBox {
  switch (direction) {
    case 'left': return [after[0], before[1], before[0], before[3]]
    case 'right': return [before[2], before[1], after[2], before[3]]
    case 'top': return [before[0], after[1], before[2], before[1]]
    case 'bottom': return [before[0], before[3], before[2], after[3]]
  }
}

interface BackgroundModel {
  readonly rgb: Rgb
  readonly tolerance: number
}

function estimateBackground(image: ImagePixels, boxes: readonly BBox[]): BackgroundModel | null {
  const samples: Rgb[] = []
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box.map(Math.round) as unknown as BBox
    const ring = 7
    for (let y = y1 - ring; y <= y2 + ring; y += 2) {
      for (let x = x1 - ring; x <= x2 + ring; x += 2) {
        if (containsPoint(box, x, y)) continue
        const rgb = pixelAt(image, x, y)
        if (rgb) samples.push(rgb)
      }
    }
  }
  if (samples.length < 8) return null

  const rgb = medianRgb(samples)
  const distances = samples.map(sample => colorDistance(sample, rgb)).sort((a, b) => a - b)
  const mad = distances[Math.floor(distances.length / 2)] ?? 0
  return { rgb, tolerance: clamp(22 + mad * 3, 26, 64) }
}

function backgroundStats(
  image: ImagePixels,
  bbox: BBox,
  background: BackgroundModel,
): { readonly samples: number; readonly coverage: number } {
  const area = bboxWidth(bbox) * bboxHeight(bbox)
  const step = clamp(Math.floor(Math.sqrt(area / 1600)), 2, 6)
  let samples = 0
  let matches = 0

  for (let y = Math.floor(bbox[1]); y < Math.ceil(bbox[3]); y += step) {
    for (let x = Math.floor(bbox[0]); x < Math.ceil(bbox[2]); x += step) {
      const rgb = pixelAt(image, x, y)
      if (!rgb) continue
      samples += 1
      if (colorDistance(rgb, background.rgb) <= background.tolerance) matches += 1
    }
  }

  return { samples, coverage: samples ? matches / samples : 0 }
}

function roundedRectBBox(rect: FitRect, pageSize: readonly [number, number]): BBox {
  const bbox = rectBBox(rect)
  return [
    Math.max(0, Math.floor(bbox[0])),
    Math.max(0, Math.floor(bbox[1])),
    Math.min(pageSize[0], Math.ceil(bbox[2])),
    Math.min(pageSize[1], Math.ceil(bbox[3])),
  ]
}

function pixelAt(image: ImagePixels, x: number, y: number): Rgb | null {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return null
  const i = (Math.floor(y) * image.width + Math.floor(x)) * 4
  return [image.data[i] ?? 255, image.data[i + 1] ?? 255, image.data[i + 2] ?? 255]
}

function colorDistance(a: Rgb, b: Rgb): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

function containsPoint(bbox: BBox, x: number, y: number): boolean {
  return bbox[0] <= x && x <= bbox[2] && bbox[1] <= y && y <= bbox[3]
}

function intersects(a: BBox, b: BBox): boolean {
  return Math.max(a[0], b[0]) < Math.min(a[2], b[2]) && Math.max(a[1], b[1]) < Math.min(a[3], b[3])
}

function sameBBox(a: BBox, b: BBox): boolean {
  return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3]
}

function medianRgb(samples: readonly Rgb[]): Rgb {
  return [median(samples.map(rgb => rgb[0])), median(samples.map(rgb => rgb[1])), median(samples.map(rgb => rgb[2]))]
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}

function bboxWidth(bbox: BBox): number {
  return Math.max(0, bbox[2] - bbox[0])
}

function bboxHeight(bbox: BBox): number {
  return Math.max(0, bbox[3] - bbox[1])
}

function bboxArea(bbox: BBox): number {
  return Math.max(1, bboxWidth(bbox) * bboxHeight(bbox))
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}
