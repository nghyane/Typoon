import type { TextPlacement } from '../domain/planning'
import type { TranslatedUnit } from '../domain/translation'
import { hasReliableBackgroundFill, type SafeMarginsDebug } from './backgroundFit'
import { fitPageText, type CssFitResult } from './fit'
import { MANGA_FONT_PROFILE } from './font'
import type { RenderLanguageContext } from './languageProfile'
import { classifyTextScript, type TextScript } from './textScript'
import { buildTextStyle, type TextStylePlan } from './textStyle'

export interface TextLayerItem {
  readonly placement: TextPlacement
  readonly unit: TranslatedUnit
}

export interface TextLayerOptions {
  readonly placementMargins?: readonly SafeMarginsDebug[]
  readonly fontContextPlacements?: readonly TextPlacement[]
  readonly languageContext?: RenderLanguageContext
}

export type FittedTextLayerItem<T extends TextLayerItem = TextLayerItem> = T & {
  readonly fit: CssFitResult
}

export function fitTextLayerItems<T extends TextLayerItem>(
  items: readonly T[],
  pageSize: readonly [number, number],
  options?: TextLayerOptions,
): Array<FittedTextLayerItem<T>> {
  const fits = fitPageText(
    items.map(item => ({ placement: item.placement, text: item.unit.targetText, sourceText: item.unit.sourceText })),
    pageSize,
    MANGA_FONT_PROFILE,
    options?.placementMargins,
    options?.fontContextPlacements,
    options?.languageContext,
  )
  return items
    .map((item, index) => ({ ...item, fit: fits[index]! }))
    .filter(renderableFittedItem)
}

function renderableFittedItem(item: FittedTextLayerItem): boolean {
  if (item.fit.overflow) return false
  if (sameNormalizedText(item.unit.sourceText, item.unit.targetText)) return false
  if (item.placement.role !== 'sfx' && item.fit.expansion && !hasReliableBackgroundFill(item.fit.expansion)) return false
  if (item.placement.role !== 'sfx') return true
  return scriptFamily(classifyTextScript(item.unit.sourceText)) === scriptFamily(classifyTextScript(item.unit.targetText))
}

function sameNormalizedText(a: string, b: string): boolean {
  const left = a.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
  const right = b.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
  return !!left && left === right
}

function scriptFamily(script: TextScript): 'latin' | 'hangul' | 'cjk' | 'other' {
  if (script === 'latin') return 'latin'
  if (script === 'hangul') return 'hangul'
  if (script === 'han' || script === 'kana' || script === 'mixed-cjk') return 'cjk'
  return 'other'
}

export function createTextLayer(
  items: readonly TextLayerItem[],
  pageSize: readonly [number, number],
  options?: TextLayerOptions,
): HTMLElement {
  return createTextLayerFromFits(fitTextLayerItems(items, pageSize, options), pageSize)
}

export function createTextLayerFromFits(
  items: readonly FittedTextLayerItem[],
  pageSize: readonly [number, number],
): HTMLElement {
  const layer = document.createElement('div')
  layer.style.position = 'absolute'
  layer.style.inset = '0'
  layer.style.pointerEvents = 'none'
  layer.style.zIndex = '1'

  for (const item of items) layer.appendChild(createTextBox(item, item.fit, pageSize))
  return layer
}

function createTextBox(
  item: TextLayerItem,
  fit: CssFitResult,
  pageSize: readonly [number, number],
): HTMLElement {
  const { placement, unit } = item
  const style = buildTextStyle(placement, fit.fontSizePx)
  const layout = placement.layoutHint
  const textAlign = layout.inlineAlign
  const justifyContent = layout.inlineAlign === 'left' ? 'flex-start' : 'center'
  const vertical = fit.direction === 'vertical'
  const [pageW, pageH] = pageSize
  const { x, y, width, height, rotationDeg } = fit.rect
  const box = document.createElement('div')
  box.dataset.typoonText = 'true'
  box.dataset.placementId = placement.id
  box.dataset.role = placement.role
  box.dataset.fontSizePx = String(fit.fontSizePx)
  box.dataset.paddingXPx = String(fit.paddingXPx)
  box.dataset.paddingYPx = String(fit.paddingYPx)
  box.dataset.maxDomFitPx = String(fit.maxDomFitPx)
  box.dataset.capReason = fit.capReason
  box.dataset.sourceFontPx = fit.sourceFontPx === null ? '' : String(fit.sourceFontPx)
  box.dataset.roleMedianFontPx = fit.roleMedianFontPx === null ? '' : String(fit.roleMedianFontPx)
  box.dataset.targetFontPx = String(fit.targetFontPx)
  box.dataset.fontIntentReason = fit.fontIntentReason
  box.dataset.fitReason = fit.fitReason
  box.dataset.overflow = String(fit.overflow)
  box.dataset.rotationDeg = rotationDeg.toFixed(2)
  box.dataset.lineCount = String(fit.lineCount)
  box.dataset.lineScore = fit.lineScore.toFixed(2)
  box.dataset.maxFill = fit.maxFill.toFixed(3)
  box.dataset.edgeGuardPx = String(fit.edgeGuardPx)
  box.dataset.fontShortSideRatio = fit.fontShortSideRatio.toFixed(3)
  box.dataset.safeShapeUsed = String(fit.safeShapeUsed)
  box.dataset.baseRect = formatRect(fit.baseRect)
  box.dataset.desiredFontSizePx = String(fit.desiredFontSizePx)
  box.dataset.layoutDirection = layout.direction
  box.dataset.direction = fit.direction
  box.dataset.directionReason = fit.directionReason
  box.dataset.layoutCandidate = fit.layoutCandidate
  box.dataset.layoutInlineAlign = layout.inlineAlign
  box.dataset.layoutBlockAlign = layout.blockAlign
  box.dataset.layoutKind = layout.kind
  box.dataset.layoutConfidence = layout.confidence.toFixed(2)
  box.dataset.layoutReason = layout.reason
  if (fit.expansion) {
    box.dataset.expansionReason = fit.expansion.reasons.overall
    box.dataset.safeBounds = formatBBox(fit.expansion.safeBounds)
    box.dataset.margins = `${fit.expansion.margins.top},${fit.expansion.margins.right},${fit.expansion.margins.bottom},${fit.expansion.margins.left}`
    box.dataset.backgroundRgb = fit.expansion.backgroundRgb?.join(',') ?? ''
    box.dataset.backgroundTolerance = String(Math.round(fit.expansion.backgroundTolerance))
    box.dataset.shapeConfidence = fit.expansion.shape ? fit.expansion.shape.confidence.toFixed(2) : ''
    box.dataset.shapeSpans = fit.expansion.shape ? String(fit.expansion.shape.spans.length) : ''
  }
  box.style.position = 'absolute'
  box.style.left = `${(x / pageW) * 100}%`
  box.style.top = `${(y / pageH) * 100}%`
  box.style.width = `${(width / pageW) * 100}%`
  box.style.height = `${(height / pageH) * 100}%`
  box.style.transformOrigin = 'center center'
  if (Math.abs(rotationDeg) > 0.1) box.style.transform = `rotate(${rotationDeg}deg)`
  box.style.display = 'flex'
  box.style.alignItems = 'center'
  box.style.justifyContent = justifyContent
  box.style.textAlign = textAlign
  box.style.padding = `calc(${fit.paddingYPx}px * var(--typoon-page-scale, 1)) calc(${fit.paddingXPx}px * var(--typoon-page-scale, 1))`
  box.style.boxSizing = 'border-box'
  box.style.overflow = 'hidden'
  box.style.color = style.fill
  box.style.textShadow = 'none'
  box.title = `${unit.sourceText}\n→\n${unit.targetText}\nfont=${fit.fontSizePx}px source=${fit.sourceFontPx ?? 'n/a'}px target=${fit.targetFontPx}px median=${fit.roleMedianFontPx ?? 'n/a'}px reason=${fit.fontIntentReason}/${fit.fitReason} layout=${fit.layoutCandidate} direction=${fit.direction}/${fit.directionReason} maxDom=${fit.maxDomFitPx}px cap=${fit.capReason} expand=${fit.expansion?.reasons.overall ?? 'none'} rot=${rotationDeg.toFixed(1)}°`

  const text = document.createElement('span')
  text.dataset.typoonTextContent = 'true'
  text.style.display = 'block'
  text.style.width = vertical ? 'auto' : '100%'
  if (vertical) text.style.height = '100%'
  text.style.position = 'relative'
  text.style.textAlign = textAlign
  text.style.filter = style.shadow ?? 'none'
  text.style.fontWeight = style.fontWeight
  text.style.fontFamily = MANGA_FONT_PROFILE.cssFamily
  text.style.fontSize = `calc(${fit.fontSizePx}px * var(--typoon-page-scale, 1))`
  text.style.lineHeight = `calc(${fit.lineHeightPx}px * var(--typoon-page-scale, 1))`
  text.style.whiteSpace = 'pre'
  text.style.overflowWrap = 'normal'
  text.style.wordBreak = 'normal'
  if (vertical) {
    text.style.writingMode = 'vertical-rl'
    text.style.textOrientation = 'mixed'
  }
  appendStyledText(text, fit.text, style)
  box.appendChild(text)
  return box
}

function formatRect(rect: { readonly x: number; readonly y: number; readonly width: number; readonly height: number }): string {
  return [rect.x, rect.y, rect.width, rect.height].map(n => n.toFixed(1)).join(',')
}

function formatBBox(bbox: readonly [number, number, number, number]): string {
  return bbox.map(n => n.toFixed(1)).join(',')
}

function appendStyledText(root: HTMLElement, value: string, style: TextStylePlan): void {
  root.style.color = style.fill
  const stroke = style.strokes[0]
  if (stroke) {
    root.style.setProperty('-webkit-text-stroke-width', `calc(${stroke.widthPx}px * var(--typoon-page-scale, 1))`)
    root.style.setProperty('-webkit-text-stroke-color', stroke.color)
    root.style.setProperty('paint-order', 'stroke fill')
  }
  root.textContent = value
}
