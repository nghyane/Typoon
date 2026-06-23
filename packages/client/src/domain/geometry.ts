export type BBox = readonly [number, number, number, number]
export type Point = readonly [number, number]
export type Polygon = readonly Point[]

/**
 * Oriented (rotated) rectangle in page-pixel space.  Unlike BBox, `w`/`h` are
 * the true unrotated extents, so a tilted text line keeps its real font size
 * instead of the inflated axis-aligned height (`w·sinθ + h·cosθ`).  This is the
 * geometry the OCR engine actually returns; keeping it avoids lossy measurement
 * downstream.
 */
export interface OrientedBox {
  readonly cx: number
  readonly cy: number
  readonly w: number
  readonly h: number
  readonly rotationDeg: number
}

export function orientedFromBBox(bbox: BBox, rotationDeg = 0): OrientedBox {
  return {
    cx: (bbox[0] + bbox[2]) / 2,
    cy: (bbox[1] + bbox[3]) / 2,
    w: Math.max(0, bbox[2] - bbox[0]),
    h: Math.max(0, bbox[3] - bbox[1]),
    rotationDeg,
  }
}

/** Overlap ratio of two 1-D intervals (centers separated by `shift`), normalized by the smaller extent. */
export function overlapRatio1D(shift: number, extentA: number, extentB: number): number {
  const inter = Math.min(Math.min(extentA, extentB), (extentA + extentB) / 2 - Math.abs(shift))
  return Math.max(0, inter) / Math.max(1, Math.min(extentA, extentB))
}

export interface OrientedPairFrame {
  /** Gap between box edges along the line-stacking axis (perpendicular to reading). */
  readonly primaryGapPx: number
  /** Overlap ratio along the reading axis. */
  readonly secondaryOverlap: number
  /** Center shift along the reading axis. */
  readonly centerShiftPx: number
  /** Larger reading-axis extent of the pair. */
  readonly maxSecondarySpan: number
  /** Font size = extent perpendicular to reading. */
  readonly fontPx: number
}

/**
 * Measure a pair of oriented boxes in their shared rotated frame.  For
 * horizontal text the reading axis is the box local-x (`u`) and lines stack
 * along local-y (`v`); for vertical text the axes swap.  Measuring here instead
 * of on axis-aligned bboxes keeps gaps/overlaps correct when text is tilted.
 */
export function orientedPairFrame(a: OrientedBox, b: OrientedBox, direction: 'horizontal' | 'vertical'): OrientedPairFrame {
  const theta = ((a.rotationDeg + b.rotationDeg) / 2) * Math.PI / 180
  const ux = Math.cos(theta)
  const uy = Math.sin(theta)
  const dx = b.cx - a.cx
  const dy = b.cy - a.cy
  const du = dx * ux + dy * uy
  const dv = -dx * uy + dy * ux

  if (direction === 'vertical') {
    return {
      primaryGapPx: Math.max(0, Math.abs(du) - (a.w + b.w) / 2),
      secondaryOverlap: overlapRatio1D(dv, a.h, b.h),
      centerShiftPx: Math.abs(dv),
      maxSecondarySpan: Math.max(a.h, b.h),
      fontPx: (a.w + b.w) / 2,
    }
  }
  return {
    primaryGapPx: Math.max(0, Math.abs(dv) - (a.h + b.h) / 2),
    secondaryOverlap: overlapRatio1D(du, a.w, b.w),
    centerShiftPx: Math.abs(du),
    maxSecondarySpan: Math.max(a.w, b.w),
    fontPx: (a.h + b.h) / 2,
  }
}

export function area(b: BBox): number {
  return Math.max(1, (b[2] - b[0]) * (b[3] - b[1]))
}

export function center(b: BBox): Point {
  return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
}

export function containsCenter(outer: BBox, inner: BBox): boolean {
  const [cx, cy] = center(inner)
  return outer[0] <= cx && cx <= outer[2] && outer[1] <= cy && cy <= outer[3]
}

export function containment(inner: BBox, outer: BBox): number {
  const ix1 = Math.max(inner[0], outer[0])
  const iy1 = Math.max(inner[1], outer[1])
  const ix2 = Math.min(inner[2], outer[2])
  const iy2 = Math.min(inner[3], outer[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  return ((ix2 - ix1) * (iy2 - iy1)) / area(inner)
}

export function iou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  return inter / (area(a) + area(b) - inter)
}

export function union(bboxes: readonly BBox[]): BBox {
  if (bboxes.length === 0) throw new Error('union requires at least one bbox')
  let x1 = Infinity
  let y1 = Infinity
  let x2 = -Infinity
  let y2 = -Infinity
  for (const b of bboxes) {
    x1 = Math.min(x1, b[0])
    y1 = Math.min(y1, b[1])
    x2 = Math.max(x2, b[2])
    y2 = Math.max(y2, b[3])
  }
  return [x1, y1, x2, y2]
}

export function clipBBox(bbox: BBox, pageSize: readonly [number, number]): BBox {
  const [w, h] = pageSize
  return [
    Math.max(0, bbox[0]),
    Math.max(0, bbox[1]),
    Math.min(w, bbox[2]),
    Math.min(h, bbox[3]),
  ]
}

export function bboxToPolygon(b: BBox): Polygon {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

export function polygonBBox(polygon: Polygon, pageSize: readonly [number, number]): BBox {
  const xs = polygon.map(p => p[0])
  const ys = polygon.map(p => p[1])
  return clipBBox([
    Math.floor(Math.min(...xs)),
    Math.floor(Math.min(...ys)),
    Math.ceil(Math.max(...xs)),
    Math.ceil(Math.max(...ys)),
  ], pageSize)
}

export function inscribedEllipse(bbox: BBox, vertices = 24): Polygon {
  const [x1, y1, x2, y2] = bbox
  const cx = (x1 + x2) / 2
  const cy = (y1 + y2) / 2
  const rx = (x2 - x1) / 2
  const ry = (y2 - y1) / 2
  const out: Point[] = []
  for (let i = 0; i < vertices; i++) {
    const a = (Math.PI * 2 * i) / vertices
    out.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)])
  }
  return out
}
