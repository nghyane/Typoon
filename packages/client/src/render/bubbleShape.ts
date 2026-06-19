import type { Polygon } from '../domain/geometry'
import type { FitRect } from './fitGeometry'

export type BubbleShapeKind = 'rect' | 'oval' | 'polygon' | 'tall' | 'wide'

export interface BubbleShapeProfile {
  readonly kind: BubbleShapeKind
  readonly centerX: number
  readonly centerY: number
  readonly rect: FitRect
  /** Max pixel width available for line `lineIndex` of `totalLines`. */
  widthAt(lineIndex: number, totalLines: number): number
}

const MIN_RATIO = 0.38

export function bubbleShapeProfile(polygon: Polygon, rect: FitRect): BubbleShapeProfile {
  const aspect = rect.width / Math.max(1, rect.height)
  const kind = polygon.length === 4 ? 'rect' : aspect < 0.72 ? 'tall' : aspect > 2.2 ? 'wide' : 'polygon'
  return {
    kind,
    centerX: rect.x + rect.width / 2,
    centerY: rect.y + rect.height / 2,
    rect,
    widthAt: (lineIndex, totalLines) => widthAtLine(polygon, rect, kind, lineIndex, totalLines),
  }
}

function widthAtLine(
  polygon: Polygon,
  rect: FitRect,
  kind: BubbleShapeKind,
  lineIndex: number,
  totalLines: number,
): number {
  if (totalLines <= 0) return rect.width

  const ratio = kind === 'polygon'
    ? polygonWidthRatio(polygon, rect, lineIndex, totalLines)
    : edgeRatio(kind, lineIndex, totalLines)

  return Math.max(1, rect.width * clamp(ratio, MIN_RATIO, 1))
}

function edgeRatio(kind: BubbleShapeKind, lineIndex: number, totalLines: number): number {
  if (kind === 'rect') return 1
  if (totalLines <= 1) return 1
  const [edge, middle] = kind === 'wide' ? [0.72, 0.98]
    : kind === 'tall' ? [0.50, 0.86]
    : [0.48, 0.96] // oval default
  const t = (lineIndex + 0.5) / totalLines
  return edge + (middle - edge) * Math.sin(t * Math.PI)
}

function polygonWidthRatio(
  polygon: Polygon,
  rect: FitRect,
  lineIndex: number,
  totalLines: number,
): number {
  const usableTop = rect.y + rect.height * 0.12
  const usableHeight = rect.height * 0.76
  const y = usableTop + ((lineIndex + 0.5) / Math.max(1, totalLines)) * usableHeight
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
