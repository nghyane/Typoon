// reader/pageProject.ts — map canvas-space placements to page-source / seam-local
// space and decide each placement's render target by geometry.
//
// Canvas layout (source px * captureScale), top→bottom:
//   [0 .. haloTop)            top halo (prev page bottom strip)
//   [haloTop .. haloTop+sh)   page N core
//   [haloTop+sh .. end)       bottom halo (next page top strip)
//
// canvas → page-N source:  sx = cx/scale,  sy = cy/scale - haloTop
// Ownership: centroid must lie in the core band, else a neighbor owns it (drop).
// Render target by bbox: inside core → page; spills below → seam-below; spills
// above → seam-above.

import type { BBox, Polygon } from '../domain/geometry'
import type { TextPlacement } from '../domain/planning'
import type { PageScanUnit } from '../domain/pageScan'
import type { SafeMarginsDebug, SafeShapeProfile } from '../render/backgroundFit'

export type RenderTarget = 'drop' | 'page' | 'seam-below' | 'seam-above'

export interface CanvasGeometry {
  readonly captureScale: number
  readonly haloTopPx: number   // page-N source px
}

/** Decide where a canvas-space placement renders, by centroid + bbox spill. */
export function routePlacement(canvasBBox: BBox, unit: PageScanUnit, geo: CanvasGeometry): RenderTarget {
  const coreTop = geo.haloTopPx * geo.captureScale
  const coreBottom = (geo.haloTopPx + unit.source.height) * geo.captureScale
  const centroidY = (canvasBBox[1] + canvasBBox[3]) / 2
  if (centroidY < coreTop || centroidY >= coreBottom) return 'drop'
  if (canvasBBox[3] > coreBottom && unit.nextIndex !== null) return 'seam-below'
  if (canvasBBox[1] < coreTop && unit.prevIndex !== null) return 'seam-above'
  return 'page'
}

/** Transform a canvas-space placement into page-N source space. */
export function canvasPlacementToSource(placement: TextPlacement, unit: PageScanUnit, geo: CanvasGeometry): TextPlacement {
  const pageSize: readonly [number, number] = [unit.source.width, unit.source.height]
  return {
    ...placement,
    pageIndex: unit.pageIndex,
    pageSize,
    bbox: canvasToSourceBBox(placement.bbox, geo),
    drawable: canvasToSourcePolygon(placement.drawable, geo),
    textBoxes: placement.textBoxes.map(box => canvasToSourceBBox(box, geo)),
    wordBoxes: placement.wordBoxes.map(box => canvasToSourceBBox(box, geo)),
  }
}

export function canvasMarginToSource(margin: SafeMarginsDebug, geo: CanvasGeometry): SafeMarginsDebug {
  const inv = 1 / geo.captureScale
  return {
    ...margin,
    margins: {
      top: margin.margins.top * inv,
      bottom: margin.margins.bottom * inv,
      left: margin.margins.left * inv,
      right: margin.margins.right * inv,
    },
    safeBounds: canvasToSourceBBox(margin.safeBounds, geo),
    componentBBox: margin.componentBBox ? canvasToSourceBBox(margin.componentBBox, geo) : null,
    shape: margin.shape ? canvasToSourceShape(margin.shape, geo) : null,
  }
}

/** Shift a page-source placement into seam-local space (origin at bridge top). */
export function shiftPlacementY(placement: TextPlacement, dy: number): TextPlacement {
  if (dy === 0) return placement
  return {
    ...placement,
    bbox: shiftBBoxY(placement.bbox, dy),
    drawable: placement.drawable.map(([x, y]) => [x, y + dy] as const) as unknown as Polygon,
    textBoxes: placement.textBoxes.map(box => shiftBBoxY(box, dy)),
    wordBoxes: placement.wordBoxes.map(box => shiftBBoxY(box, dy)),
  }
}

export function shiftMarginY(margin: SafeMarginsDebug, dy: number): SafeMarginsDebug {
  if (dy === 0) return margin
  return {
    ...margin,
    safeBounds: shiftBBoxY(margin.safeBounds, dy),
    componentBBox: margin.componentBBox ? shiftBBoxY(margin.componentBBox, dy) : null,
    shape: margin.shape
      ? { ...margin.shape, bounds: shiftBBoxY(margin.shape.bounds, dy), spans: margin.shape.spans.map(s => ({ ...s, y: s.y + dy })) }
      : null,
  }
}

function canvasToSourceBBox(bbox: BBox, geo: CanvasGeometry): BBox {
  const inv = 1 / geo.captureScale
  return [
    bbox[0] * inv,
    bbox[1] * inv - geo.haloTopPx,
    bbox[2] * inv,
    bbox[3] * inv - geo.haloTopPx,
  ]
}

function canvasToSourcePolygon(polygon: Polygon, geo: CanvasGeometry): Polygon {
  const inv = 1 / geo.captureScale
  return polygon.map(([x, y]) => [x * inv, y * inv - geo.haloTopPx]) as unknown as Polygon
}

function canvasToSourceShape(shape: SafeShapeProfile, geo: CanvasGeometry): SafeShapeProfile {
  const inv = 1 / geo.captureScale
  return {
    ...shape,
    bounds: canvasToSourceBBox(shape.bounds, geo),
    spans: shape.spans.map(span => ({ y: span.y * inv - geo.haloTopPx, x1: span.x1 * inv, x2: span.x2 * inv })),
  }
}

function shiftBBoxY(bbox: BBox, dy: number): BBox {
  return [bbox[0], bbox[1] + dy, bbox[2], bbox[3] + dy]
}
