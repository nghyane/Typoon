import type { Polygon } from '../domain/geometry'
import type { FitRect } from './fitGeometry'

export type BubbleShapeKind = 'rect' | 'oval' | 'polygon' | 'tall' | 'wide'

export interface BubbleShapeProfile {
  readonly kind: BubbleShapeKind
  readonly centerX: number
  readonly centerY: number
  readonly rect: FitRect
  /**
   * Max pixel width available for line `lineIndex` of `totalLines`.
   * `contentFraction` is the text block's height as a fraction of the rect
   * (≤1): a short block is vertically CENTRED in a tall bubble, so its lines
   * sit in the wide middle band, not at the narrow top/bottom of the contour.
   * Defaults to 1 (block fills the rect) for callers that don't track it.
   */
  widthAt(lineIndex: number, totalLines: number, contentFraction?: number): number
}

/** Centre a `totalLines`-line block of relative height `contentFraction` in the rect. */
export function centeredLineFraction(lineIndex: number, totalLines: number, contentFraction = 1): number {
  if (totalLines <= 0) return 0.5
  const cf = clamp(contentFraction, 0, 1)
  return 0.5 + cf * ((lineIndex + 0.5) / totalLines - 0.5)
}

const MIN_RATIO = 0.38

export function bubbleShapeProfile(polygon: Polygon, rect: FitRect, rounded = false): BubbleShapeProfile {
  const aspect = rect.width / Math.max(1, rect.height)
  // A 4-point drawable carries no contour. For a rounded speech bubble (no
  // detected shape to fall back on) treat it as an oval so lines near the top/
  // bottom narrow instead of filling the bbox corners and spilling past the
  // outline. Rectangular captions keep 'rect'.
  const tallWide = aspect < 0.72 ? 'tall' : aspect > 2.2 ? 'wide' : null
  const kind: BubbleShapeKind = polygon.length === 4
    ? (rounded ? (tallWide ?? 'oval') : 'rect')
    : (tallWide ?? 'polygon')
  return {
    kind,
    centerX: rect.x + rect.width / 2,
    centerY: rect.y + rect.height / 2,
    rect,
    widthAt: (lineIndex, totalLines, contentFraction) => widthAtLine(polygon, rect, kind, lineIndex, totalLines, contentFraction),
  }
}

function widthAtLine(
  polygon: Polygon,
  rect: FitRect,
  kind: BubbleShapeKind,
  lineIndex: number,
  totalLines: number,
  contentFraction = 1,
): number {
  if (totalLines <= 0) return rect.width

  const ratio = kind === 'polygon'
    ? polygonWidthRatio(polygon, rect, lineIndex, totalLines, contentFraction)
    : edgeRatio(kind, lineIndex, totalLines, contentFraction)

  return Math.max(1, rect.width * clamp(ratio, MIN_RATIO, 1))
}

function edgeRatio(kind: BubbleShapeKind, lineIndex: number, totalLines: number, contentFraction = 1): number {
  if (kind === 'rect') return 1
  if (totalLines <= 1) return 1
  const [edge, middle] = kind === 'wide' ? [0.72, 0.98]
    : kind === 'tall' ? [0.50, 0.86]
    : [0.48, 0.96] // oval default
  const t = centeredLineFraction(lineIndex, totalLines, contentFraction)
  return edge + (middle - edge) * Math.sin(t * Math.PI)
}

function polygonWidthRatio(
  polygon: Polygon,
  rect: FitRect,
  lineIndex: number,
  totalLines: number,
  contentFraction = 1,
): number {
  const usableTop = rect.y + rect.height * 0.12
  const usableHeight = rect.height * 0.76
  const y = usableTop + centeredLineFraction(lineIndex, totalLines, contentFraction) * usableHeight
  const xs: number[] = []
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]!
    const b = polygon[(i + 1) % polygon.length]!
    const [x1, y1] = a
    const [x2, y2] = b
    if (y1 === y2) continue
    const minY = Math.min(y1, y2)
    const maxY = Math.max(y1, y2)
    if (y < minY || y >= maxY) continue
    const t = (y - y1) / (y2 - y1)
    xs.push(x1 + (x2 - x1) * t)
  }
  xs.sort((a, b) => a - b)
  if (xs.length < 2) return edgeRatio('oval', lineIndex, totalLines)
  const width = Math.max(0, xs[xs.length - 1]! - xs[0]!)
  return width / Math.max(1, rect.width)
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}
