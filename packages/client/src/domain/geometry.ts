export type BBox = readonly [number, number, number, number]
export type Point = readonly [number, number]
export type Polygon = readonly Point[]

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
