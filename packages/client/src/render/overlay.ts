import type { TextPlacement } from '../domain/planning'
import type { TranslatedUnit } from '../domain/translation'
import type { SafeMarginsDebug } from './backgroundFit'
import { createDebugLayer, type OverlayDebugOptions } from './debugLayer'
import { createEraseLayer } from './eraseLayer'
import { buildErasePlan, type EraseStrategy } from './erasePlan'
import type { RenderLanguageContext } from './languageProfile'
import { createTextLayerFromFits, fitTextLayerItems } from './textLayer'

type TextRenderItem = {
  readonly placement: TextPlacement
  readonly unit: TranslatedUnit
  readonly margin: SafeMarginsDebug | undefined
}

export interface OverlayOptions {
  readonly eraseStrategy?: EraseStrategy
  readonly debug?: OverlayDebugOptions
}

export interface OverlayRenderData {
  readonly placements: readonly TextPlacement[]
  readonly translations: readonly TranslatedUnit[]
  readonly pageSize: readonly [number, number]
  readonly placementMargins?: readonly SafeMarginsDebug[]
  readonly fontContextPlacements?: readonly TextPlacement[]
  readonly sourceLanguage?: string | null
  readonly targetLanguage?: string | null
}

export function attachOverlay(
  host: HTMLElement,
  data: OverlayRenderData,
  options: OverlayOptions = {},
): HTMLElement {
  const style = getComputedStyle(host)
  if (style.position === 'static') host.style.position = 'relative'
  const overlay = createOverlayElement(data, options)
  host.appendChild(overlay)
  return overlay
}

export function createOverlayElement(
  data: OverlayRenderData,
  options: OverlayOptions = {},
): HTMLElement {
  const root = document.createElement('div')
  root.dataset.typoonOverlay = 'true'
  root.style.position = 'absolute'
  root.style.inset = '0'
  root.style.pointerEvents = 'none'
  root.style.zIndex = '1'

  const byUnitId = new Map(data.translations.map(unit => [unit.unitId, unit]))
  const textItems = data.placements
    .map((placement, index) => ({ placement, unit: translatedUnitForPlacement(placement, byUnitId), margin: data.placementMargins?.[index] }))
    .filter((item): item is TextRenderItem => !!item.unit && item.unit.kind !== 'skip' && item.unit.targetText.trim() !== '')
  const textMargins = alignedTextMargins(textItems)
  const languageContext: RenderLanguageContext = {
    sourceLanguage: data.sourceLanguage,
    targetLanguage: data.targetLanguage,
  }
  const fittedTextItems = fitTextLayerItems(textItems, data.pageSize, {
    placementMargins: textMargins,
    fontContextPlacements: data.fontContextPlacements,
    languageContext,
  })
  const fittedMargins = alignedTextMargins(fittedTextItems)

  root.appendChild(createEraseLayer(
    buildErasePlan(fittedTextItems.map(item => item.placement), {
      strategy: options.eraseStrategy,
      placementMargins: fittedMargins,
    }),
    data.pageSize,
  ))
  root.appendChild(createTextLayerFromFits(fittedTextItems, data.pageSize))
  if (hasDebugOptions(options.debug)) root.appendChild(createDebugLayer(data.placements, data.pageSize, options.debug))
  return root
}

function alignedTextMargins(
  items: readonly { readonly margin?: SafeMarginsDebug }[],
): readonly SafeMarginsDebug[] | undefined {
  if (!items.length) return undefined
  const margins = items.map(item => item.margin)
  return margins.every((margin): margin is SafeMarginsDebug => !!margin) ? margins : undefined
}

function translatedUnitForPlacement(placement: TextPlacement, byUnitId: ReadonlyMap<string, TranslatedUnit>): TranslatedUnit | null {
  const units = placement.sourceUnitIds.map(id => byUnitId.get(id)).filter((unit): unit is TranslatedUnit => !!unit)
  if (!units.length) return null
  return {
    unitId: units.map(unit => unit.unitId).join('+'),
    pageIndex: placement.pageIndex,
    kind: units.some(unit => unit.kind !== 'skip') ? placement.role === 'sfx' ? 'sfx' : 'dialogue' : 'skip',
    role: placement.role,
    sourceText: units.map(unit => unit.sourceText).join('\n'),
    targetText: units.map(unit => unit.targetText).filter(Boolean).join('\n'),
  }
}

function hasDebugOptions(options: OverlayDebugOptions | undefined): boolean {
  return !!options && Object.values(options).some(Boolean)
}
