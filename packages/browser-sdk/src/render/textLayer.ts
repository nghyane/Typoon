import type { ImagePixels } from '../domain/image'
import type { TextPlacement } from '../domain/planning'
import type { TranslatedUnit } from '../domain/translation'
import { fitPageText } from './fit'
import { MANGA_FONT_PROFILE } from './font'
import { buildTextStyle, type TextStylePlan, type TextStrokePlan } from './textStyle'

export interface TextLayerItem {
  readonly placement: TextPlacement
  readonly unit: TranslatedUnit
}

export function createTextLayer(items: readonly TextLayerItem[], pageSize: readonly [number, number], image?: ImagePixels): HTMLElement {
  const [pageW] = pageSize
  const layer = document.createElement('div')
  layer.style.position = 'absolute'
  layer.style.inset = '0'
  layer.style.pointerEvents = 'none'

  const fits = fitPageText(items.map(item => ({ placement: item.placement, text: item.unit.targetText })), pageW, MANGA_FONT_PROFILE)
  for (let i = 0; i < items.length; i++) {
    layer.appendChild(createTextBox(items[i]!, fits[i]!, pageSize, image))
  }
  return layer
}

function createTextBox(
  item: TextLayerItem,
  fit: ReturnType<typeof fitPageText>[number],
  pageSize: readonly [number, number],
  image?: ImagePixels,
): HTMLElement {
  const { placement, unit } = item
  const style = buildTextStyle(placement, image)
  const [pageW, pageH] = pageSize
  const { x, y, width, height, rotationDeg } = fit.rect
  const box = document.createElement('div')
  box.dataset.typoonText = 'true'
  box.dataset.placementId = placement.id
  box.dataset.role = placement.role
  box.dataset.fontSizePx = String(fit.fontSizePx)
  box.dataset.maxDomFitPx = String(fit.maxDomFitPx)
  box.dataset.capReason = fit.capReason
  box.dataset.overflow = String(fit.overflow)
  box.dataset.rotationDeg = rotationDeg.toFixed(2)
  box.style.position = 'absolute'
  box.style.left = `${(x / pageW) * 100}%`
  box.style.top = `${(y / pageH) * 100}%`
  box.style.width = `${(width / pageW) * 100}%`
  box.style.height = `${(height / pageH) * 100}%`
  box.style.transformOrigin = 'center center'
  if (Math.abs(rotationDeg) > 0.1) box.style.transform = `rotate(${rotationDeg}deg)`
  box.style.display = 'flex'
  box.style.alignItems = 'center'
  box.style.justifyContent = 'center'
  box.style.textAlign = 'center'
  box.style.padding = '0'
  box.style.boxSizing = 'border-box'
  box.style.overflow = 'hidden'
  box.style.color = style.fill
  box.style.textShadow = 'none'
  box.title = `${placement.sourceText}\n→\n${unit.targetText}\nfont=${fit.fontSizePx}px maxDom=${fit.maxDomFitPx}px cap=${fit.capReason} rot=${rotationDeg.toFixed(1)}°`

  const text = document.createElement('span')
  text.dataset.typoonTextContent = 'true'
  text.style.display = 'block'
  text.style.width = '100%'
  text.style.position = 'relative'
  text.style.filter = style.shadow ?? 'none'
  text.style.fontWeight = style.fontWeight
  text.style.fontFamily = MANGA_FONT_PROFILE.cssFamily
  text.style.fontSize = `calc(${fit.fontSizePx}px * var(--typoon-page-scale, 1))`
  text.style.lineHeight = `calc(${fit.lineHeightPx}px * var(--typoon-page-scale, 1))`
  text.style.whiteSpace = 'normal'
  text.style.overflowWrap = 'anywhere'
  text.style.wordBreak = 'normal'
  appendStyledText(text, fit.text, style)
  box.appendChild(text)
  return box
}

function appendStyledText(root: HTMLElement, value: string, style: TextStylePlan): void {
  if (!style.strokes.length) {
    root.style.color = style.fill
    root.textContent = value
    return
  }

  root.style.color = style.fill
  style.strokes.forEach((stroke, index) => {
    const layer = createTextStrokeLayer(value, style.fill, stroke)
    layer.style.position = 'absolute'
    layer.style.inset = '0'
    layer.style.zIndex = String(index)
    root.appendChild(layer)
  })

  const fill = document.createElement('span')
  fill.style.position = 'relative'
  fill.style.zIndex = String(style.strokes.length)
  fill.style.display = 'block'
  fill.style.color = style.fill
  fill.textContent = value
  root.appendChild(fill)
}

function createTextStrokeLayer(value: string, fill: string, stroke: TextStrokePlan): HTMLElement {
  const layer = document.createElement('span')
  layer.style.display = 'block'
  layer.style.width = '100%'
  layer.style.color = fill
  layer.style.setProperty('-webkit-text-stroke-width', `calc(${stroke.widthPx}px * var(--typoon-page-scale, 1))`)
  layer.style.setProperty('-webkit-text-stroke-color', stroke.color)
  layer.textContent = value
  return layer
}
