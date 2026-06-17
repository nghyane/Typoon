import type { ImagePixels } from '../domain/image'
import type { TextPlacement } from '../domain/planning'
import type { TranslatedUnit } from '../domain/translation'
import { createDebugLayer, type OverlayDebugOptions } from './debugLayer'
import { createEraseLayer } from './eraseLayer'
import { buildErasePlan, type EraseStrategy } from './erasePlan'
import { createTextLayer } from './textLayer'

export interface OverlayOptions {
  readonly eraseStrategy?: EraseStrategy
  readonly debug?: OverlayDebugOptions
}

export function attachOverlay(
  host: HTMLElement,
  result: {
    placements: readonly TextPlacement[]
    translations: readonly TranslatedUnit[]
    pageSize: readonly [number, number]
    image?: ImagePixels
  },
  options: OverlayOptions = {},
): HTMLElement {
  const style = getComputedStyle(host)
  if (style.position === 'static') host.style.position = 'relative'
  const overlay = createOverlayElement(result, options)
  host.appendChild(overlay)
  bindOverlayScale(overlay, result.pageSize)
  return overlay
}

export function createOverlayElement(
  result: {
    placements: readonly TextPlacement[]
    translations: readonly TranslatedUnit[]
    pageSize: readonly [number, number]
    image?: ImagePixels
  },
  options: OverlayOptions = {},
): HTMLElement {
  const root = document.createElement('div')
  root.dataset.typoonOverlay = 'true'
  root.style.position = 'absolute'
  root.style.inset = '0'
  root.style.pointerEvents = 'none'
  root.style.overflow = 'hidden'
  root.style.setProperty('--typoon-page-scale', '1')

  const byUnitId = new Map(result.translations.map(unit => [unit.unitId, unit]))
  const textItems = result.placements
    .map(placement => ({ placement, unit: translatedUnitForPlacement(placement, byUnitId) }))
    .filter((item): item is { placement: TextPlacement; unit: TranslatedUnit } => !!item.unit && item.unit.kind !== 'skip' && item.unit.targetText.trim() !== '')

  root.appendChild(createEraseLayer(buildErasePlan(result.placements, result.image, { strategy: options.eraseStrategy }), result.pageSize))
  root.appendChild(createTextLayer(textItems, result.pageSize, result.image))
  if (hasDebugOptions(options.debug)) root.appendChild(createDebugLayer(result.placements, result.pageSize, options.debug))
  return root
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

function bindOverlayScale(overlay: HTMLElement, pageSize: readonly [number, number]): void {
  const update = (): void => {
    const rect = overlay.getBoundingClientRect()
    const sx = rect.width / pageSize[0]
    const sy = rect.height / pageSize[1]
    const scale = Math.min(sx, sy)
    overlay.style.setProperty('--typoon-page-scale', Number.isFinite(scale) && scale > 0 ? String(scale) : '1')
  }
  update()
  if (!('ResizeObserver' in window)) return
  const observer = new ResizeObserver(update)
  observer.observe(overlay)
  const remove = overlay.remove.bind(overlay)
  overlay.remove = () => {
    observer.disconnect()
    remove()
  }
}
