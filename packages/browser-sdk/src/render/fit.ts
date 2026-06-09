import type { Polygon } from '../domain/geometry'
import type { FontHint, TextPlacement } from '../domain/planning'
import type { FontProfile } from './font'

const MIN_FONT_SIZE = 8
const ABS_MAX_FONT_SIZE = 96
const MAX_FONT_PAGE_FRACTION = 0.05
const PAGE_BODY_GROWTH = 1.20
const HINT_OUTLIER_MIN = 10
const HINT_OUTLIER_MAX = 60
const DEFAULT_INSET = 2
const ELLIPSE_FIT_SCALE = 0.85

export interface CssFitInput {
  readonly placement: TextPlacement
  readonly text: string
}

export interface CssFitResult {
  readonly text: string
  readonly fontSizePx: number
  readonly lineHeightPx: number
  readonly overflow: boolean
  readonly rect: FitRect
  readonly maxDomFitPx: number
  readonly capReason: string
}

export interface FitRect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
  readonly rotationDeg: number
}

export function fitPageText(items: readonly CssFitInput[], pageWidth: number, font: FontProfile): CssFitResult[] {
  const measurer = createDomMeasurer(font)
  try {
    const pageCap = pageBodyCap(items.map(item => item.placement.fontHint), maxFontForPage(pageWidth))
    return items.map(item => fitPlacementText(item, pageCap, font, measurer))
  } finally {
    measurer.destroy()
  }
}

function fitPlacementText(input: CssFitInput, pageCap: number, font: FontProfile, measurer: DomMeasurer): CssFitResult {
  const text = normalizeText(input.text)
  const rect = drawableRect(input.placement)
  const safeW = rect.width
  const safeH = rect.height
  if (!text || safeW < 1 || safeH < 1) {
    return {
      text,
      fontSizePx: MIN_FONT_SIZE,
      lineHeightPx: MIN_FONT_SIZE * font.lineHeightRatio,
      overflow: false,
      rect,
      maxDomFitPx: MIN_FONT_SIZE,
      capReason: 'empty',
    }
  }

  const style = styleForPlacement(input.placement)
  const maxDomFitPx = maxFittingSize({
    text,
    width: safeW,
    height: safeH,
    hiBound: Math.max(MIN_FONT_SIZE, Math.min(Math.floor(safeH), ABS_MAX_FONT_SIZE)),
    fontWeight: style.fontWeight,
    measurer,
  })
  const caps = [
    { reason: 'dom', px: maxDomFitPx },
    { reason: 'page-body', px: pageCap },
    { reason: 'source-font', px: sourceFontCap(input.placement) },
    { reason: 'role-geometry', px: roleGeometryCap(input.placement, safeW, safeH) },
  ]
  const limit = caps.reduce((best, cap) => cap.px < best.px ? cap : best)
  let bestSize = Math.max(MIN_FONT_SIZE, Math.floor(limit.px))

  while (bestSize > MIN_FONT_SIZE && !measurer.fits({ text, width: safeW, height: safeH, fontSizePx: bestSize, fontWeight: style.fontWeight })) {
    bestSize -= 1
  }

  const overflow = !measurer.fits({ text, width: safeW, height: safeH, fontSizePx: bestSize, fontWeight: style.fontWeight })
  return {
    text,
    fontSizePx: bestSize,
    lineHeightPx: bestSize * font.lineHeightRatio,
    overflow,
    rect,
    maxDomFitPx,
    capReason: limit.reason,
  }
}

function maxFittingSize(args: {
  readonly text: string
  readonly width: number
  readonly height: number
  readonly hiBound: number
  readonly fontWeight: string
  readonly measurer: DomMeasurer
}): number {
  let lo = MIN_FONT_SIZE
  let hi = args.hiBound
  let best = MIN_FONT_SIZE
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2)
    if (args.measurer.fits({
      text: args.text,
      width: args.width,
      height: args.height,
      fontSizePx: mid,
      fontWeight: args.fontWeight,
    })) {
      best = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }
  return best
}

interface CssTextStyle {
  readonly fontWeight: string
}

function styleForPlacement(placement: TextPlacement): CssTextStyle {
  return { fontWeight: placement.role === 'sfx' ? '800' : '700' }
}

interface MeasureRequest {
  readonly text: string
  readonly width: number
  readonly height: number
  readonly fontSizePx: number
  readonly fontWeight: string
}

interface DomMeasurer {
  fits(request: MeasureRequest): boolean
  destroy(): void
}

function createDomMeasurer(font: FontProfile): DomMeasurer {
  const el = document.createElement('div')
  el.style.position = 'fixed'
  el.style.left = '-10000px'
  el.style.top = '0'
  el.style.visibility = 'hidden'
  el.style.pointerEvents = 'none'
  el.style.boxSizing = 'border-box'
  el.style.display = 'block'
  el.style.textAlign = 'center'
  el.style.whiteSpace = 'normal'
  el.style.overflowWrap = 'anywhere'
  el.style.wordBreak = 'normal'
  el.style.fontFamily = font.cssFamily
  el.style.padding = '0'
  document.body.appendChild(el)

  return {
    fits(request: MeasureRequest): boolean {
      el.style.width = `${request.width}px`
      el.style.height = 'auto'
      el.style.fontSize = `${request.fontSizePx}px`
      el.style.lineHeight = `${request.fontSizePx * font.lineHeightRatio}px`
      el.style.fontWeight = request.fontWeight
      el.textContent = request.text

      const rect = el.getBoundingClientRect()
      return el.scrollWidth <= Math.ceil(request.width) && rect.height <= request.height + 0.5
    },
    destroy(): void {
      el.remove()
    },
  }
}

function drawableRect(placement: TextPlacement): FitRect {
  if (Math.abs(placement.rotationDeg) > 0.1 && polygonAngleDeg(placement.drawable) === 0) {
    return rotatedAabbRect(placement.bbox, placement.rotationDeg)
  }
  if (placement.drawable.length === 4) return orientedRect(placement.drawable)
  return axisAlignedRect(placement.drawable)
}

function polygonAngleDeg(polygon: Polygon): number {
  const p0 = polygon[0]
  const p1 = polygon[1]
  if (!p0 || !p1) return 0
  const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]) * 180 / Math.PI
  return Math.abs(angle) > 0.1 ? angle : 0
}

function rotatedAabbRect(bbox: readonly [number, number, number, number], rotationDeg: number): FitRect {
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
  const w = rawW * scale
  const h = rawH * scale
  return {
    x: x1 + DEFAULT_INSET + (rawW - w) / 2,
    y: y1 + DEFAULT_INSET + (rawH - h) / 2,
    width: w,
    height: h,
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

function maxFontForPage(pageWidth: number): number {
  return clamp(Math.floor(pageWidth * MAX_FONT_PAGE_FRACTION), 48, ABS_MAX_FONT_SIZE)
}

function pageBodyCap(hints: readonly (FontHint | null)[], pageGeometryMax: number): number {
  const samples = hints
    .map(h => h?.sourceFontPx ?? 0)
    .filter(s => HINT_OUTLIER_MIN <= s && s <= HINT_OUTLIER_MAX)
    .sort((a, b) => a - b)
  if (!samples.length) return pageGeometryMax
  const median = samples[Math.floor(samples.length / 2)]!
  return Math.min(pageGeometryMax, Math.max(MIN_FONT_SIZE, Math.round(median * PAGE_BODY_GROWTH)))
}

function sourceFontCap(placement: TextPlacement): number {
  const fontSize = placement.fontHint?.sourceFontPx ?? 0
  if (fontSize < HINT_OUTLIER_MIN || fontSize > HINT_OUTLIER_MAX) return ABS_MAX_FONT_SIZE
  return Math.max(MIN_FONT_SIZE, Math.round(fontSize * sourceGrowth(placement.role)))
}

function sourceGrowth(role: TextPlacement['role']): number {
  switch (role) {
    case 'sfx': return 1.35
    case 'dialogue': return 1.22
    case 'narration': return 1.15
  }
}

function roleGeometryCap(placement: TextPlacement, safeW: number, safeH: number): number {
  const shortSide = Math.min(safeW, safeH)
  const fraction = placement.role === 'sfx' ? 0.55 : placement.role === 'dialogue' ? 0.28 : 0.24
  return Math.max(MIN_FONT_SIZE, Math.round(shortSide * fraction))
}

function normalizeText(text: string): string {
  return text
    .split(/\r?\n/u)
    .map(line => line.split(/\s+/u).filter(Boolean).join(' '))
    .filter(Boolean)
    .join(' ')
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}
