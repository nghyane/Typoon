import type { ErasePlan, EraseShape } from './erasePlan'

export function createEraseLayer(plan: ErasePlan, pageSize: readonly [number, number]): SVGSVGElement {
  const [pageW, pageH] = pageSize
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('viewBox', `0 0 ${pageW} ${pageH}`)
  svg.setAttribute('preserveAspectRatio', 'none')
  svg.style.position = 'absolute'
  svg.style.inset = '0'
  svg.style.width = '100%'
  svg.style.height = '100%'
  svg.style.pointerEvents = 'none'

  if (plan.kind === 'flat-fill') {
    for (const shape of plan.shapes) svg.appendChild(createShapeElement(shape))
  }
  return svg
}

function createShapeElement(shape: EraseShape): SVGElement {
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  rect.setAttribute('x', String(shape.cx - shape.width / 2))
  rect.setAttribute('y', String(shape.cy - shape.height / 2))
  rect.setAttribute('width', String(shape.width))
  rect.setAttribute('height', String(shape.height))
  rect.setAttribute('rx', String(shape.radius))
  rect.setAttribute('fill', shape.fill)
  if (Math.abs(shape.rotationDeg) > 0.1) {
    rect.setAttribute('transform', `rotate(${shape.rotationDeg} ${shape.cx} ${shape.cy})`)
  }
  return rect
}
