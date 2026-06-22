// reader/pagePipeline.ts — page-local stage composition.
//   capture (core + halo) → OCR → detect → noise filter → place → translate
//   → route by geometry → page-source / seam-local overlay.
//
// All recognition/placement/margin work happens in capture-canvas space (so the
// background-fit estimator sees matching pixels); results are transformed to
// page-source or seam-local space only at the end.

import type { PageScanUnit, ReaderPageOverlay, SeamOverlay, PlacementItem } from '../domain/pageScan'
import { emptyReaderPageOverlay } from '../domain/pageScan'
import type { TextPlacement } from '../domain/planning'
import type { TranslationUnit, TranslatedUnit } from '../domain/translation'
import type { RecognizedTextPage } from '../domain/text'
import type { TextRegion } from '../domain/regions'
import type { SafeMarginsDebug } from '../render/backgroundFit'
import { estimateSafeMargins } from '../render/backgroundFit'
import { textFitRect } from '../render/fitGeometry'
import { LensTextRecognizer } from '../recognizers/lens/LensTextRecognizer'
import { buildOverlayPlacements } from '../pipeline/composeOverlay'
import { textFromRecognition, translatePreparedText, type PreparedTextResult } from '../pipeline/translatePreparedPage'
import { removeReaderNoiseBlocks } from '../pipeline/readerNoise'
import { textRoleContext, type TextRoleContext } from '../pipeline/textRole'
import type { Translator } from '../translators/translator'
import type { BBox } from '../domain/geometry'
import { capturePageScan, type CapturedPageScan } from './pageCapture'
import {
  routePlacement,
  canvasPlacementToSource,
  canvasMarginToSource,
  shiftPlacementY,
  shiftMarginY,
  type CanvasGeometry,
} from './pageProject'
import { detectTextRegions } from './visionRuntime'
import type { LoadedPage } from './pageProvider'
import type { TranslationConfig } from './translationConfig'
import { throwIfAborted } from './asyncSignal'

export interface PagePipelineDeps {
  readonly recognizer: LensTextRecognizer
  readonly translator: () => Translator
  readonly config: TranslationConfig
}

export interface PagePipelineArgs {
  readonly unit: PageScanUnit
  readonly loadPage: (index: number) => Promise<LoadedPage>
  readonly sourceLanguage: string | null
  readonly targetLanguage: string
  readonly signal: AbortSignal
}

interface RoutedItem {
  readonly placement: TextPlacement
  readonly margin: SafeMarginsDebug
}

interface CoreFrame {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

interface ScannedPage {
  readonly capture: CapturedPageScan
  readonly clean: RecognizedTextPage
  readonly regions: readonly TextRegion[]
  readonly coreFrame: CoreFrame
  readonly roleContext: TextRoleContext
}

interface ComposedPage {
  readonly text: PreparedTextResult
  readonly renderPlacements: readonly TextPlacement[]
  readonly translationInput: readonly TranslationUnit[]
}

export class PagePipeline {
  constructor(private readonly deps: PagePipelineDeps) {}

  async run(args: PagePipelineArgs): Promise<ReaderPageOverlay> {
    const { unit, signal } = args
    const empty = emptyReaderPageOverlay(unit.pageIndex, unit.source)

    const scanned = await this.scanPage(args)
    const composed = this.composePlacements(scanned, unit.pageIndex)
    if (!composed) return empty

    const translations = await this.translatePlacements(composed, args)
    throwIfAborted(signal)

    const margins = estimateCanvasMargins(scanned.capture.image, composed.renderPlacements)
    const geo: CanvasGeometry = { captureScale: scanned.capture.captureScale, haloTopPx: scanned.capture.haloTopPx }
    return this.route(unit, composed.renderPlacements, margins, geo, translations)
  }

  /** Capture core+halo, OCR, detect regions, drop noise, derive role context. */
  private async scanPage(args: PagePipelineArgs): Promise<ScannedPage> {
    const { unit, signal } = args
    const capture = await capturePageScan(unit, args.loadPage, this.deps.config.scan, signal)
    throwIfAborted(signal)
    const recognized = await this.deps.recognizer.recognizeEncoded(capture.encoded, {
      pageIndex: unit.pageIndex,
      sourceLang: args.sourceLanguage,
      signal,
    })
    throwIfAborted(signal)
    const regions = await detectTextRegions(capture.image, signal, this.deps.config)
    throwIfAborted(signal)

    const coreFrame: CoreFrame = {
      x: 0,
      y: capture.haloTopPx * capture.captureScale,
      width: capture.image.width,
      height: unit.source.height * capture.captureScale,
    }
    const clean = removeReaderNoiseBlocks(recognized, coreFrame)
    const coreOwnedBlocks = clean.blocks.filter(block => {
      const cy = (block.bbox[1] + block.bbox[3]) / 2
      return cy >= coreFrame.y && cy < coreFrame.y + coreFrame.height
    })
    const roleContext: TextRoleContext = coreOwnedBlocks.length ? textRoleContext(coreOwnedBlocks) : {}
    return { capture, clean, regions, coreFrame, roleContext }
  }

  /** Build overlay placements + synthetic translation units; null if nothing to do. */
  private composePlacements(scanned: ScannedPage, pageIndex: number): ComposedPage | null {
    const { clean, regions, roleContext } = scanned
    const text = textFromRecognition({ pageIndex, recognized: clean, regions, roleContext })
    const placements = buildOverlayPlacements({ recognized: text.recognized, textUnits: text.textUnits, regions, roleContext })
    if (!placements.length) return null

    const translationInput = translationInputFromPlacements(placements, text.translationUnits)
    if (!hasTranslatableUnit(translationInput)) return null
    const renderPlacements = placements.map(p => ({ ...p, sourceUnitIds: renderSourceUnitIds(p) }))
    return { text, renderPlacements, translationInput }
  }

  /** Translate the composed units and return the translated payloads. */
  private async translatePlacements(composed: ComposedPage, args: PagePipelineArgs): Promise<readonly TranslatedUnit[]> {
    const translated = await translatePreparedText({
      text: { ...composed.text, textUnits: [], translationUnits: composed.translationInput },
      translator: this.deps.translator(),
      sourceLanguage: args.sourceLanguage,
      targetLanguage: args.targetLanguage,
      signal: args.signal,
    })
    return translated.translations
  }

  private route(
    unit: PageScanUnit,
    placements: readonly TextPlacement[],
    margins: readonly SafeMarginsDebug[],
    geo: CanvasGeometry,
    translations: readonly TranslatedUnit[],
  ): ReaderPageOverlay {
    const pageItems: RoutedItem[] = []
    const seamBelowItems: RoutedItem[] = []
    const seamAboveItems: RoutedItem[] = []

    for (let i = 0; i < placements.length; i += 1) {
      const placement = placements[i]!
      const margin = margins[i]!
      const target = routePlacement(placement.bbox, unit, geo)
      if (target === 'drop') continue

      const sourcePlacement = canvasPlacementToSource(placement, unit, geo)
      const sourceMargin = canvasMarginToSource(margin, geo)

      if (target === 'page') {
        pageItems.push({ placement: sourcePlacement, margin: sourceMargin })
      } else if (target === 'seam-below') {
        const seamSize = [unit.source.width, unit.source.height + unit.haloBottomPx] as const
        seamBelowItems.push({
          placement: { ...sourcePlacement, pageSize: seamSize },
          margin: sourceMargin,
        })
      } else {
        const dy = unit.haloTopPx
        const seamSize = [unit.source.width, unit.haloTopPx + unit.source.height] as const
        seamAboveItems.push({
          placement: { ...shiftPlacementY(sourcePlacement, dy), pageSize: seamSize },
          margin: shiftMarginY(sourceMargin, dy),
        })
      }
    }

    return {
      pageIndex: unit.pageIndex,
      pageSize: [unit.source.width, unit.source.height],
      items: toPlacementItems(pageItems),
      translations: filterTranslations(translations, pageItems),
      seamBelow: buildSeam(seamBelowItems, translations, {
        topPageIndex: unit.pageIndex,
        bottomPageIndex: unit.nextIndex ?? unit.pageIndex,
        seamSize: [unit.source.width, unit.source.height + unit.haloBottomPx],
        seamSplitY: unit.source.height,
      }),
      seamAbove: buildSeam(seamAboveItems, translations, {
        topPageIndex: unit.prevIndex ?? unit.pageIndex,
        bottomPageIndex: unit.pageIndex,
        seamSize: [unit.source.width, unit.haloTopPx + unit.source.height],
        seamSplitY: unit.haloTopPx,
      }),
    }
  }
}

function buildSeam(
  items: readonly RoutedItem[],
  translations: readonly TranslatedUnit[],
  meta: Pick<SeamOverlay, 'topPageIndex' | 'bottomPageIndex' | 'seamSize' | 'seamSplitY'>,
): SeamOverlay | null {
  if (!items.length) return null
  return {
    ...meta,
    items: toPlacementItems(items),
    translations: filterTranslations(translations, items),
  }
}

function toPlacementItems(items: readonly RoutedItem[]): readonly PlacementItem[] {
  return items.map(item => ({ placement: item.placement, margin: item.margin }))
}

function filterTranslations(translations: readonly TranslatedUnit[], items: readonly RoutedItem[]): readonly TranslatedUnit[] {
  const referenced = new Set(items.flatMap(item => item.placement.sourceUnitIds))
  return translations.filter(unit => referenced.has(unit.unitId))
}

function estimateCanvasMargins(
  image: { readonly width: number; readonly height: number; readonly data: Uint8ClampedArray },
  placements: readonly TextPlacement[],
): readonly SafeMarginsDebug[] {
  const pageSize: readonly [number, number] = [image.width, image.height]
  return placements.map((placement, index) => {
    const baseRect = textFitRect(placement)
    const obstacles = placements.filter((_, i) => i !== index).flatMap(placementBBoxes)
    return estimateSafeMargins({ image, placement, baseRect, obstacles, pageSize })
  })
}

function placementBBoxes(placement: TextPlacement): readonly BBox[] {
  return placement.textBoxes.length ? placement.textBoxes : [placement.bbox]
}

// ── synthetic per-placement translation units (clean, no strip deps) ──

function translationInputFromPlacements(
  placements: readonly TextPlacement[],
  units: readonly TranslationUnit[],
): readonly TranslationUnit[] {
  const byId = new Map(units.map(unit => [unit.id, unit]))
  return placements.map(placement => {
    const sourceUnits = placement.sourceUnitIds.map(id => byId.get(id)).filter((unit): unit is TranslationUnit => !!unit)
    const translatableUnits = sourceUnits.filter(unit => unit.kind !== 'skip' && unit.sourceText.trim())
    const sourceText = translatableUnits.map(unit => unit.sourceText).filter(Boolean).join('\n')
    const skipped = translatableUnits.length === 0
    return {
      id: placementUnitId(placement),
      pageIndex: placement.pageIndex,
      blockIds: translatableUnits.flatMap(unit => unit.blockIds),
      sourceText,
      kind: skipped || !sourceText.trim() ? 'skip' : placement.role === 'sfx' ? 'sfx' : 'dialogue',
      role: placement.role,
    }
  })
}

function placementUnitId(placement: TextPlacement): string {
  return `${placement.id}-unit`
}

function renderSourceUnitIds(placement: TextPlacement): readonly string[] {
  const syntheticId = placementUnitId(placement)
  return placement.sourceUnitIds.length > 1 ? [syntheticId, ...placement.sourceUnitIds] : [syntheticId]
}

function hasTranslatableUnit(units: readonly { readonly kind: string; readonly sourceText: string }[]): boolean {
  return units.some(unit => unit.kind !== 'skip' && unit.sourceText.trim())
}
