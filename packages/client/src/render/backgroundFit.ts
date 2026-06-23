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
const SHAPE_SCAN_MIN_STEP_PX = 3
const SHAPE_SCAN_MAX_STEP_PX = 8
const MIN_SHAPE_SPANS = 3

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
  readonly shape: SafeShapeProfile | null
}

export interface SafeShapeSpan {
  readonly y: number
  readonly x1: number
  readonly x2: number
}

export interface SafeShapeProfile {
  readonly bounds: BBox
  readonly spans: readonly SafeShapeSpan[]
  readonly confidence: number
}

export function hasReliableBackgroundFill(margin: SafeMarginsDebug | null | undefined): margin is SafeMarginsDebug & { readonly backgroundRgb: Rgb } {
  return !!margin?.backgroundRgb && margin.componentConfidence >= 0.6
}

export function hasAnyBackgroundFill(margin: SafeMarginsDebug | null | undefined): margin is SafeMarginsDebug & { readonly backgroundRgb: Rgb } {
  return !!margin?.backgroundRgb
}

export function estimateSafeMargins(args: {
  readonly image: ImagePixels
  readonly placement: TextPlacement
  readonly baseRect: FitRect
  readonly obstacles: readonly BBox[]
  readonly pageSize: readonly [number, number]
}): SafeMarginsDebug {
  const rotated = Math.abs(args.baseRect.rotationDeg) > 0.1
  // For a tilted rect the unrotated rectBBox is not its screen footprint; use
  // the placement's axis-aligned bbox (true tilted extent) as the seed instead.
  const base = rotated
    ? clampBBoxToPage(args.placement.bbox, args.pageSize)
    : roundedRectBBox(args.baseRect, args.pageSize)

  // SFX is rendered as free oriented text with no bubble fill, so a rotated SFX
  // box must not drive flood-fill.  Tilted dialogue/narration (e.g. chat bubbles
  // on a phone screen) still sit on a solid background that needs filling, so
  // background estimation must run for them using the axis-aligned seed above.
  if (rotated && args.placement.role === 'sfx') {
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
  const shape = estimateSafeShapeProfile({
    image: args.image,
    bounds: component.bbox,
    seed: base,
    selfBoxes,
    obstacles: args.obstacles,
    background,
    confidence: component.confidence,
  })

  return {
    reasons,
    margins,
    safeBounds: component.bbox,
    backgroundRgb: background.rgb,
    backgroundTolerance: background.tolerance,
    componentBBox: component.bbox,
    componentConfidence: component.confidence,
    shape,
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
    shape: null,
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

      if (args.obstacles.some(obstacle => obstacleBlocks(strip, obstacle))) continue

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
  const coverageStats = backgroundStats(args.image, bounds, args.background, args.selfBoxes)
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

function estimateSafeShapeProfile(args: {
  readonly image: ImagePixels
  readonly bounds: BBox
  readonly seed: BBox
  readonly selfBoxes: readonly BBox[]
  readonly obstacles: readonly BBox[]
  readonly background: BackgroundModel
  readonly confidence: number
}): SafeShapeProfile | null {
  if (args.confidence < 0.6) return null
  const height = bboxHeight(args.bounds)
  if (height < 4 || bboxWidth(args.bounds) < 4) return null

  const step = clamp(Math.floor(height / 36), SHAPE_SCAN_MIN_STEP_PX, SHAPE_SCAN_MAX_STEP_PX)
  const y1 = Math.floor(args.bounds[1])
  const y2 = Math.ceil(args.bounds[3])
  const spans: SafeShapeSpan[] = []
  let expectedRows = 0

  for (let y = y1; y <= y2; y += step) {
    expectedRows += 1
    const span = scanShapeSpan(args, y)
    if (span) spans.push(span)
  }

  if (spans.length < MIN_SHAPE_SPANS) return null
  const coverage = spans.length / Math.max(1, expectedRows)
  return { bounds: args.bounds, spans, confidence: args.confidence * coverage }
}

function scanShapeSpan(args: {
  readonly image: ImagePixels
  readonly bounds: BBox
  readonly seed: BBox
  readonly selfBoxes: readonly BBox[]
  readonly obstacles: readonly BBox[]
  readonly background: BackgroundModel
}, y: number): SafeShapeSpan | null {
  const x1 = Math.floor(args.bounds[0])
  const x2 = Math.ceil(args.bounds[2])
  const runs: Array<{ readonly x1: number; readonly x2: number }> = []
  let start: number | null = null

  for (let x = x1; x <= x2; x += 1) {
    const ok = isShapePixel(args, x, y)
    if (ok && start === null) start = x
    if ((!ok || x === x2) && start !== null) {
      const end = ok && x === x2 ? x + 1 : x
      if (end - start >= 2) runs.push({ x1: start, x2: end })
      start = null
    }
  }

  const run = pickShapeRun(runs, args.seed)
  return run ? { y, x1: run.x1, x2: run.x2 } : null
}

function isShapePixel(args: {
  readonly image: ImagePixels
  readonly selfBoxes: readonly BBox[]
  readonly obstacles: readonly BBox[]
  readonly background: BackgroundModel
}, x: number, y: number): boolean {
  if (args.selfBoxes.some(box => containsPoint(box, x, y))) return true
  if (args.obstacles.some(box => containsPoint(box, x, y))) return false
  const rgb = pixelAt(args.image, x, y)
  return !!rgb && colorDistance(rgb, args.background.rgb) <= args.background.tolerance
}

function pickShapeRun(
  runs: readonly { readonly x1: number; readonly x2: number }[],
  seed: BBox,
): { readonly x1: number; readonly x2: number } | null {
  if (!runs.length) return null
  const seedCx = (seed[0] + seed[2]) / 2
  const containing = runs.filter(run => run.x1 <= seedCx && seedCx <= run.x2)
  if (containing.length) return widestRun(containing)
  const overlapping = runs
    .map(run => ({ run, overlap: Math.max(0, Math.min(run.x2, seed[2]) - Math.max(run.x1, seed[0])) }))
    .filter(item => item.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap || runWidth(b.run) - runWidth(a.run))
  return overlapping[0]?.run ?? widestRun(runs)
}

function widestRun(runs: readonly { readonly x1: number; readonly x2: number }[]): { readonly x1: number; readonly x2: number } | null {
  let best = runs[0]
  if (!best) return null
  for (const run of runs.slice(1)) {
    if (runWidth(run) > runWidth(best)) best = run
  }
  return best
}

function runWidth(run: { readonly x1: number; readonly x2: number }): number {
  return Math.max(0, run.x2 - run.x1)
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
  skipBoxes?: readonly BBox[],
): { readonly samples: number; readonly coverage: number } {
  const area = bboxWidth(bbox) * bboxHeight(bbox)
  const step = clamp(Math.floor(Math.sqrt(area / 1600)), 2, 6)
  let samples = 0
  let matches = 0

  for (let y = Math.floor(bbox[1]); y < Math.ceil(bbox[3]); y += step) {
    for (let x = Math.floor(bbox[0]); x < Math.ceil(bbox[2]); x += step) {
      if (skipBoxes?.some(box => containsPoint(box, x, y))) continue
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

function clampBBoxToPage(bbox: BBox, pageSize: readonly [number, number]): BBox {
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

/** Only block expansion if the obstacle covers more than 30% of the strip.
 *  Corner-only touches from adjacent bubbles are ignored. */
function obstacleBlocks(strip: BBox, obstacle: BBox): boolean {
  const ix1 = Math.max(strip[0], obstacle[0])
  const iy1 = Math.max(strip[1], obstacle[1])
  const ix2 = Math.min(strip[2], obstacle[2])
  const iy2 = Math.min(strip[3], obstacle[3])
  if (ix1 >= ix2 || iy1 >= iy2) return false
  const stripArea = bboxWidth(strip) * bboxHeight(strip)
  return (ix2 - ix1) * (iy2 - iy1) > stripArea * 0.3
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
