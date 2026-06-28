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
import { recoverBubbleText, type BubbleCropRecognizer, type BubbleSource } from '../pipeline/bubbleRecovery'
import { textFromRecognition, translatePreparedText, type PreparedTextResult } from '../pipeline/translatePreparedPage'
import { removeReaderNoiseBlocks } from '../pipeline/readerNoise'
import { removeOcrArtifactBlocks } from '../pipeline/ocrArtifacts'
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
    // OCR (Lens, network) and region detection (ONNX, inference) both consume the
    // same capture image and are independent — run them concurrently so the page
    // critical path is max(OCR, detect) instead of OCR + detect.
    const [recognizedRaw, regions] = await Promise.all([
      this.deps.recognizer.recognizeText(capture.image, {
        pageIndex: unit.pageIndex,
        sourceLang: args.sourceLanguage,
        signal,
      }),
      detectTextRegions(capture.image, signal, this.deps.config),
    ])
    throwIfAborted(signal)
    const recognized = removeOcrArtifactBlocks(recognizedRaw)

    const source: BubbleSource = {
      loadFullCanvas: () => loadStitchedSourceCanvas(args.loadPage, unit, signal),
      captureScale: capture.captureScale,
    }
    const recovered = await recoverBubbleText({
      recognized,
      source,
      regions,
      recognizer: this.cropRecognizer(unit.pageIndex, args.sourceLanguage, signal),
    })
    throwIfAborted(signal)

    const coreFrame: CoreFrame = {
      x: 0,
      y: capture.haloTopPx * capture.captureScale,
      width: capture.image.width,
      height: unit.source.height * capture.captureScale,
    }
    const clean = removeReaderNoiseBlocks(recovered, coreFrame)
    const coreOwnedBlocks = clean.blocks.filter(block => {
      const cy = (block.bbox[1] + block.bbox[3]) / 2
      return cy >= coreFrame.y && cy < coreFrame.y + coreFrame.height
    })
    const roleContext: TextRoleContext = coreOwnedBlocks.length ? textRoleContext(coreOwnedBlocks) : {}
    return { capture, clean, regions, coreFrame, roleContext }
  }

  /** Adapt the recognizer to the per-bubble crop interface Phase B needs. */
  private cropRecognizer(pageIndex: number, sourceLanguage: string | null, signal: AbortSignal): BubbleCropRecognizer {
    return {
      recognizeCrop: image => this.deps.recognizer.recognizeText(image, { pageIndex, sourceLang: sourceLanguage, signal }),
    }
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
      const target = routePlacement(placement.bbox, margin, unit, geo)
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

// ── source-image loading for Phase B bubble recovery ──

async function loadStitchedSourceCanvas(
  loadPage: (index: number) => Promise<LoadedPage>,
  unit: PageScanUnit,
  signal: AbortSignal,
): Promise<HTMLCanvasElement> {
  throwIfAborted(signal)

  const totalH = unit.haloTopPx + unit.source.height + unit.haloBottomPx
  const canvas = document.createElement('canvas')
  canvas.width = unit.source.width
  canvas.height = totalH
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d canvas unavailable')
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  // Page N core at y = haloTopPx
  {
    const page = await loadPage(unit.pageIndex)
    const bitmap = await createImageBitmap(page.blob, { imageOrientation: 'from-image' })
    try {
      ctx.drawImage(bitmap, 0, unit.haloTopPx)
    } finally {
      bitmap.close()
    }
  }

  // Prev page bottom strip at y = 0
  if (unit.prevIndex !== null && unit.haloTopPx > 0) {
    throwIfAborted(signal)
    const prev = await loadPage(unit.prevIndex)
    const bitmap = await createImageBitmap(prev.blob, { imageOrientation: 'from-image' })
    try {
      const stripH = Math.min(unit.haloTopPx, prev.size.height)
      const srcY = prev.size.height - stripH
      const dx = Math.round((unit.source.width - prev.size.width) / 2)
      ctx.drawImage(bitmap, 0, srcY, prev.size.width, stripH, dx, 0, prev.size.width, stripH)
    } finally {
      bitmap.close()
    }
  }

  // Next page top strip at y = haloTopPx + pageHeight
  if (unit.nextIndex !== null && unit.haloBottomPx > 0) {
    throwIfAborted(signal)
    const next = await loadPage(unit.nextIndex)
    const bitmap = await createImageBitmap(next.blob, { imageOrientation: 'from-image' })
    try {
      const stripH = Math.min(unit.haloBottomPx, next.size.height)
      const dx = Math.round((unit.source.width - next.size.width) / 2)
      const destY = unit.haloTopPx + unit.source.height
      ctx.drawImage(bitmap, 0, 0, next.size.width, stripH, dx, destY, next.size.width, stripH)
    } finally {
      bitmap.close()
    }
  }

  return canvas
}

// ── cross-page seam deduplication ─────────────────────────────────────────
// When a bubble spans the bottom of page N and the top of page N+1, both page
// captures include it (via neighbor halos), so seam-below of page N and
// seam-above of page N+1 may both contain blocks for the same bubble.
// Deduplicate by matching source text: keep the seam-below copy, remove the
// duplicate from seam-above so the text is never rendered twice in the gap.

function normalizedSourceText(placement: TextPlacement, translations: readonly TranslatedUnit[]): string {
  const uniq = [...new Set(placement.sourceUnitIds)]
  return uniq
    .map(id => translations.find(t => t.unitId === id)?.sourceText ?? '')
    .filter(Boolean)
    .join('\n')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase()
}

/** BBox overlap ratio: intersection area / smaller area. */
function bboxOverlapRatio(a: BBox, b: BBox): number {
  const ax = Math.max(a[0], b[0])
  const ay = Math.max(a[1], b[1])
  const bx = Math.min(a[2], b[2])
  const by = Math.min(a[3], b[3])
  if (ax >= bx || ay >= by) return 0
  const inter = (bx - ax) * (by - ay)
  const areaA = (a[2] - a[0]) * (a[3] - a[1])
  const areaB = (b[2] - b[0]) * (b[3] - b[1])
  return inter / Math.min(areaA, areaB)
}

/**
 * Are these two seam copies (compared in page-N source space) the same straddling
 * bubble? The two captures OCR the bubble independently, so they wrap/place it at
 * slightly different heights and the area overlap alone can dip well below a strict
 * threshold even though it is plainly the same bubble. We therefore match on EITHER
 * a lenient area overlap OR a same-column signal: at most one bubble can straddle a
 * given seam in a given x-column, so equal width + strong x-overlap + centers within
 * a box height is a safe duplicate test (distinct side-by-side bubbles share little
 * x-overlap; distinct stacked bubbles cannot both cross the same seam line).
 */
function seamBboxDuplicate(below: BBox, above: BBox): boolean {
  if (bboxOverlapRatio(below, above) > 0.3) return true
  const wBelow = below[2] - below[0]
  const wAbove = above[2] - above[0]
  const widthRatio = Math.min(wBelow, wAbove) / Math.max(1, wBelow, wAbove)
  const xOverlap = (Math.min(below[2], above[2]) - Math.max(below[0], above[0])) / Math.max(1, Math.min(wBelow, wAbove))
  const centerGap = Math.abs((below[1] + below[3]) / 2 - (above[1] + above[3]) / 2)
  const avgHeight = ((below[3] - below[1]) + (above[3] - above[1])) / 2
  return widthRatio >= 0.6 && xOverlap >= 0.6 && centerGap <= avgHeight
}

/** Convert a bbox in seamAbove(N+1) space to page-N source space. */
function seamAboveBboxToPageNSpace(bbox: BBox, haloTopN1: number, pageNHeight: number): BBox {
  const dy = pageNHeight - haloTopN1
  return [bbox[0], bbox[1] + dy, bbox[2], bbox[3] + dy]
}

export function deduplicateSeamBlocks(overlays: Map<number, ReaderPageOverlay>): void {
  const indices = [...overlays.keys()].sort((a, b) => a - b)
  for (const pageIndex of indices) {
    const overlay = overlays.get(pageIndex)
    if (!overlay?.seamBelow) continue

    const nextOverlay = overlays.get(pageIndex + 1)
    if (!nextOverlay?.seamAbove) continue

    const belowTexts = new Set(
      overlay.seamBelow.items.map(item =>
        normalizedSourceText(item.placement, overlay.seamBelow!.translations),
      ),
    )

    const belowBboxes = overlay.seamBelow.items.map(item => item.placement.bbox)

    const filtered = nextOverlay.seamAbove.items.filter(item => {
      const text = normalizedSourceText(item.placement, nextOverlay.seamAbove!.translations)
      if (text && belowTexts.has(text)) return false

      const aboveBboxInPageN = seamAboveBboxToPageNSpace(
        item.placement.bbox,
        nextOverlay.seamAbove!.seamSplitY,
        overlay.seamBelow!.seamSplitY,
      )
      for (const belowBbox of belowBboxes) {
        if (seamBboxDuplicate(belowBbox, aboveBboxInPageN)) return false
      }

      return true
    })

    if (filtered.length < nextOverlay.seamAbove.items.length) {
      const newSeamAbove: SeamOverlay | null = filtered.length === 0 ? null : {
        ...nextOverlay.seamAbove,
        items: filtered,
        translations: nextOverlay.seamAbove.translations.filter(t =>
          filtered.some(item => item.placement.sourceUnitIds.includes(t.unitId)),
        ),
      }
      overlays.set(pageIndex + 1, { ...nextOverlay, seamAbove: newSeamAbove })
    }
  }
}
