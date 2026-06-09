import type { BBox, Polygon } from '../domain/geometry'
import type { TextPlacement } from '../domain/planning'

export interface OverlayDebugOptions {
  readonly showDrawable?: boolean
  readonly showTextBoxes?: boolean
  readonly showTextBounds?: boolean
  readonly showLabels?: boolean
}

export function createDebugLayer(
  placements: readonly TextPlacement[],
  pageSize: readonly [number, number],
  options: OverlayDebugOptions = {},
): SVGSVGElement {
  const [pageW, pageH] = pageSize
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.dataset.typoonDebugLayer = 'true'
  svg.setAttribute('viewBox', `0 0 ${pageW} ${pageH}`)
  svg.setAttribute('preserveAspectRatio', 'none')
  svg.style.position = 'absolute'
  svg.style.inset = '0'
  svg.style.width = '100%'
  svg.style.height = '100%'
  svg.style.pointerEvents = 'none'

  for (const placement of placements) {
    if (options.showDrawable) svg.appendChild(createPolygon(placement.drawable, roleColor(placement.role), 3))
    if (options.showTextBoxes) {
      for (const box of placement.textBoxes) svg.appendChild(createRect(box, '#38bdf8', 2))
    }
    if (options.showTextBounds) svg.appendChild(createRect(placement.bbox, '#f97316', 2))
    if (options.showLabels) svg.appendChild(createLabel(placement))
  }

  return svg
}

function createPolygon(polygon: Polygon, color: string, width: number): SVGPolygonElement {
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
  el.setAttribute('points', polygon.map(([x, y]) => `${x},${y}`).join(' '))
  el.setAttribute('fill', 'none')
  el.setAttribute('stroke', color)
  el.setAttribute('stroke-width', String(width))
  el.setAttribute('vector-effect', 'non-scaling-stroke')
  return el
}

function createRect(bbox: BBox, color: string, width: number): SVGRectElement {
  const [x1, y1, x2, y2] = bbox
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  el.setAttribute('x', String(x1))
  el.setAttribute('y', String(y1))
  el.setAttribute('width', String(Math.max(1, x2 - x1)))
  el.setAttribute('height', String(Math.max(1, y2 - y1)))
  el.setAttribute('fill', 'none')
  el.setAttribute('stroke', color)
  el.setAttribute('stroke-width', String(width))
  el.setAttribute('vector-effect', 'non-scaling-stroke')
  return el
}

function createLabel(placement: TextPlacement): SVGTextElement {
  const [x1, y1] = placement.bbox
  const el = document.createElementNS('http://www.w3.org/2000/svg', 'text')
  el.setAttribute('x', String(x1))
  el.setAttribute('y', String(Math.max(12, y1 - 4)))
  el.setAttribute('fill', roleColor(placement.role))
  el.setAttribute('font-size', '14')
  el.setAttribute('font-family', 'monospace')
  el.setAttribute('paint-order', 'stroke')
  el.setAttribute('stroke', 'rgba(0,0,0,.8)')
  el.setAttribute('stroke-width', '3')
  el.textContent = `${placement.id} ${placement.role} rot=${placement.rotationDeg.toFixed(1)}`
  return el
}

function roleColor(role: TextPlacement['role']): string {
  switch (role) {
    case 'sfx': return '#f43f5e'
    case 'dialogue': return '#22c55e'
    case 'narration': return '#a855f7'
  }
}
