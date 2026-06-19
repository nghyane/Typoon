/**
 * Page-local translation runtime.
 *
 * Source pages are the progress source of truth:
 *   load -> identity prepare -> OCR -> translate -> compose -> overlay/done
 *
 * Cross-page OCR is a repair pass only.  A seam repair image is created after
 * both adjacent source pages are prepared, then OCR/translated best-effort and
 * projected back onto the two source pages.  It never blocks source page
 * progress, but the run waits for bounded seam tasks before completing so their
 * overlays can still attach.
 */

import type { PageTranslationStatus, TranslationProgress, TranslationRequest, TranslationRunEvent } from '../domain/run'
import type { PageOverlay } from '../domain/overlay'
import type { PageDocumentSource } from '../domain/source'
import type { PreparedPageHandle } from '../domain/prepared'
import type { BBox } from '../domain/geometry'
import type { RecognizedTextPage, TextBlock } from '../domain/text'
import type { PreparationSession, VisionRuntime } from '../vision/VisionRuntime'
import type { TextRecognizer } from '../recognizers/text'
import type { Translator } from '../translators/translator'
import { StageExecutor, type PipelineConcurrency } from './StageExecutor'
import { buildSourcePlan } from './SourceOrder'
import { preparedTextFromRecognition, translatePreparedText } from '../pipeline/translatePreparedPage'
import { buildOverlayPlacements, composeAndProjectOverlays } from '../pipeline/composeOverlay'
import { rectBBox, textFitRect } from '../render/fitGeometry'

const DEFAULT_SEAM_BAND_PX = 320
const EDGE_TEXT_WINDOW_LEFT_RATIO = 0.12
const EDGE_TEXT_WINDOW_RIGHT_RATIO = 0.92
const EDGE_TEXT_MIN_GLYPH_RATIO = 0.018
const EDGE_TEXT_MIN_GLYPH_PX = 12
const OVERLAY_GEOMETRY_TWIN_NOISE = 0.20

type SourceEdgeText = {
  readonly top: boolean
  readonly bottom: boolean
}

type SourceState = {
  handle: PreparedPageHandle | null
  edgeText: SourceEdgeText | null
  sourceSettled: boolean
  seamBeforeSettled: boolean
  seamAfterSettled: boolean
  released: boolean
}

type PreparedPageProcessResult = {
  readonly overlays: readonly PageOverlay[]
}

type EdgeTextSample = {
  readonly bbox: BBox
  readonly text: string
}

type OverlayPlacementItem = {
  readonly placement: PageOverlay['placements'][number]
  readonly margin: PageOverlay['placementMargins'][number]
}

const NO_EDGE_TEXT: SourceEdgeText = { top: false, bottom: false }

export class TranslationRun {
  private readonly runId = crypto.randomUUID()
  private readonly abort = new AbortController()
  private readonly listeners = new Set<(event: TranslationRunEvent) => void>()
  private readonly pageStatuses = new Map<number, PageTranslationStatus>()
  private readonly overlays: PageOverlay[] = []
  private lastProgress: TranslationProgress | null = null
  private readonly source: PageDocumentSource
  private readonly request: TranslationRequest
  private readonly deps: {
    readonly vision: VisionRuntime
    readonly recognizer: TextRecognizer
    readonly translator: Translator
    readonly executor: StageExecutor
    readonly concurrency: PipelineConcurrency
  }
  private started = false
  private finished = false

  private doneResolve!: (pages: readonly PageOverlay[]) => void
  private doneReject!: (error: Error) => void

  readonly done = new Promise<readonly PageOverlay[]>((resolve, reject) => {
    this.doneResolve = resolve
    this.doneReject = reject
  })

  constructor(
    source: PageDocumentSource,
    request: TranslationRequest,
    deps: {
      readonly vision: VisionRuntime
      readonly recognizer: TextRecognizer
      readonly translator: Translator
      readonly executor: StageExecutor
      readonly concurrency: PipelineConcurrency
    },
  ) {
    this.source = source
    this.request = request
    this.deps = deps
  }

  subscribe(listener: (event: TranslationRunEvent) => void): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  start(): void {
    if (this.started || this.finished) return
    this.started = true
    void this.run()
  }

  cancel(reason?: Error): void {
    if (!this.abort.signal.aborted) this.abort.abort(reason)
    this.deps.vision.cancelRun(this.runId)
    this.finishCancelled()
  }

  private async run(): Promise<void> {
    const signal = this.abort.signal
    let session: Awaited<ReturnType<VisionRuntime['beginPreparation']>> | null = null

    try {
      if (this.request.preparation.type === 'continuous-strip') {
        session = await this.deps.vision.beginPreparation(this.runId, this.request.preparation)
        await this.runContinuousStrip(session)
        if (signal.aborted) this.finishCancelled()
        else this.finishCompleted()
        return
      }

      session = await this.deps.vision.beginPreparation(this.runId, { type: 'identity' })

      const sourcePlan = buildSourcePlan({
        pageCount: this.source.pageCount,
        scope: this.request.scope ?? 'all',
        priority: this.request.priority,
        preparation: { type: 'identity' },
      })
      const activePages = new Set(sourcePlan.statusPages)
      const repairSeams = this.request.preparation.type === 'identity-with-seams' && !!this.deps.vision.createSeamRepair
      const seamBandPx = this.request.preparation.type === 'identity-with-seams'
        ? this.request.preparation.seamBandPx ?? DEFAULT_SEAM_BAND_PX
        : DEFAULT_SEAM_BAND_PX

      this.setInitialPageStatuses(sourcePlan.statusPages)
      this.emitProgress()

      const states = new Map<number, SourceState>()
      for (const pageIndex of sourcePlan.statusPages) {
        states.set(pageIndex, {
          handle: null,
          edgeText: null,
          sourceSettled: false,
          seamBeforeSettled: !repairSeams || !activePages.has(pageIndex - 1),
          seamAfterSettled: !repairSeams || !activePages.has(pageIndex + 1),
          released: false,
        })
      }

      const sourceTasks: Promise<void>[] = []
      const seamTasks: Promise<void>[] = []
      const scheduledBoundaries = new Set<number>()
      const settledBoundaries = new Set<number>()

      const releaseIfReady = (pageIndex: number): void => {
        const state = states.get(pageIndex)
        if (!state || state.released || !state.handle) return
        if (!state.sourceSettled || !state.seamBeforeSettled || !state.seamAfterSettled) return
        state.released = true
        this.deps.vision.release(state.handle)
      }

      const markBoundarySettled = (topIndex: number): void => {
        if (settledBoundaries.has(topIndex)) return
        settledBoundaries.add(topIndex)
        const top = states.get(topIndex)
        const bottom = states.get(topIndex + 1)
        if (top) top.seamAfterSettled = true
        if (bottom) bottom.seamBeforeSettled = true
        releaseIfReady(topIndex)
        releaseIfReady(topIndex + 1)
      }

      const skipPageSeams = (pageIndex: number): void => {
        const state = states.get(pageIndex)
        if (state) {
          state.edgeText = NO_EDGE_TEXT
          state.seamBeforeSettled = true
          state.seamAfterSettled = true
        }
        if (activePages.has(pageIndex - 1)) markBoundarySettled(pageIndex - 1)
        if (activePages.has(pageIndex + 1)) markBoundarySettled(pageIndex)
      }

      const evaluateBoundary = (topIndex: number): void => {
        if (!repairSeams || scheduledBoundaries.has(topIndex) || settledBoundaries.has(topIndex)) return
        const top = states.get(topIndex)
        const bottom = states.get(topIndex + 1)
        if (!top?.handle || !bottom?.handle || !top.edgeText || !bottom.edgeText) return

        if (!top.edgeText.bottom || !bottom.edgeText.top) {
          markBoundarySettled(topIndex)
          return
        }

        scheduledBoundaries.add(topIndex)
        const task = this.processSeamRepair(top.handle, bottom.handle, seamBandPx)
          .catch(() => undefined)
          .finally(() => markBoundarySettled(topIndex))
        seamTasks.push(task)
      }

      const onSourceHandleReady = (pageIndex: number, handle: PreparedPageHandle): void => {
        const state = states.get(pageIndex)
        if (!state) return
        state.handle = handle

        const task = this.processSourcePage(pageIndex, handle, seamBandPx, edgeText => {
          state.edgeText = edgeText
          evaluateBoundary(pageIndex - 1)
          evaluateBoundary(pageIndex)
        })
          .finally(() => {
            state.edgeText ??= NO_EDGE_TEXT
            state.sourceSettled = true
            evaluateBoundary(pageIndex - 1)
            evaluateBoundary(pageIndex)
            releaseIfReady(pageIndex)
          })
        sourceTasks.push(task)
      }

      await Promise.all(sourcePlan.loadOrder.map(pageIndex =>
        this.deps.executor.load(async () => {
          if (signal.aborted || this.finished) return
          this.emitStatus(pageIndex, 'loading')

          try {
            const asset = await this.source.readPage(pageIndex, signal)
            if (signal.aborted || this.finished) return

            this.emitStatus(pageIndex, 'preparing')
            const handles = await this.deps.executor.prepare(() => session!.push(asset, signal))
            const sourceHandle = handles.find(handle => handle.kind !== 'seam-repair')
            if (!sourceHandle) throw new Error(`prepare emitted no source page for ${pageIndex + 1}`)
            onSourceHandleReady(pageIndex, sourceHandle)
          } catch (error) {
            if (signal.aborted || this.finished) return
            this.markPageError(pageIndex, error)
            const state = states.get(pageIndex)
            if (state) state.sourceSettled = true
            skipPageSeams(pageIndex)
            releaseIfReady(pageIndex)
          }
        }),
      ))

      await session.flush(signal).catch(() => [])
      await Promise.allSettled(sourceTasks)
      await Promise.allSettled(seamTasks)

      for (const pageIndex of sourcePlan.statusPages) releaseIfReady(pageIndex)

      if (signal.aborted) this.finishCancelled()
      else this.finishCompleted()
    } catch (error) {
      if (signal.aborted) {
        this.finishCancelled()
        return
      }
      this.finishFailed(error instanceof Error ? error : new Error(String(error)))
    } finally {
      session?.dispose?.()
    }
  }

  private async runContinuousStrip(session: PreparationSession): Promise<void> {
    const signal = this.abort.signal
    const sourcePlan = buildSourcePlan({
      pageCount: this.source.pageCount,
      scope: this.request.scope ?? 'all',
      priority: this.request.priority,
      preparation: this.request.preparation,
    })

    this.setInitialPageStatuses(sourcePlan.statusPages)
    this.emitProgress()

    const processTasks: Promise<void>[] = []
    const liveProcessTasks = new Set<Promise<void>>()
    const maxLivePreparedPages = Math.max(1, this.deps.concurrency.maxPreparedPages ?? 2)
    let lastBufferedSourceIndex: number | null = null

    const waitForProcessCapacity = async (): Promise<void> => {
      while (liveProcessTasks.size >= maxLivePreparedPages) {
        await Promise.race(liveProcessTasks).catch(() => undefined)
      }
    }

    const processHandles = async (handles: readonly PreparedPageHandle[]): Promise<void> => {
      for (const handle of handles) {
        if (signal.aborted || this.finished) return
        await waitForProcessCapacity()
        if (signal.aborted || this.finished) return
        let trackedTask!: Promise<void>
        trackedTask = this.processContinuousPreparedPage(handle)
          .finally(() => liveProcessTasks.delete(trackedTask))
        liveProcessTasks.add(trackedTask)
        processTasks.push(trackedTask)
      }
    }

    const flushBuffered = async (): Promise<void> => {
      const handles = await this.deps.executor.prepare(() => session.flush(signal))
      lastBufferedSourceIndex = null
      await processHandles(handles)
    }

    for (const pageIndex of sourcePlan.prepareOrder) {
      if (signal.aborted || this.finished) return
      if (lastBufferedSourceIndex !== null && pageIndex !== lastBufferedSourceIndex + 1) {
        await flushBuffered()
      }

      this.emitStatus(pageIndex, 'loading')
      try {
        const asset = await this.deps.executor.load(() => this.source.readPage(pageIndex, signal))
        if (signal.aborted || this.finished) return

        this.emitStatus(pageIndex, 'preparing')
        const handles = await this.deps.executor.prepare(() => session.push(asset, signal))
        lastBufferedSourceIndex = pageIndex
        await processHandles(handles)
      } catch (error) {
        if (signal.aborted || this.finished) return
        this.markPageError(pageIndex, error)
        await flushBuffered().catch(flushError => {
          if (!signal.aborted && !this.finished) throw flushError
        })
      }
    }

    await flushBuffered()
    await Promise.allSettled(processTasks)
  }

  private async processContinuousPreparedPage(handle: PreparedPageHandle): Promise<void> {
    const pageIndexes = sourcePageIndexes(handle)
    try {
      const { overlays } = await this.processPreparedPage(handle, false)
      for (const overlay of overlays) {
        this.emitMergedOverlay(overlay)
      }

      for (const pageIndex of pageIndexes) {
        this.emitStatus(pageIndex, 'done')
      }
    } catch (error) {
      if (this.abort.signal.aborted || this.finished) return
      for (const pageIndex of pageIndexes) this.markPageError(pageIndex, error)
    } finally {
      this.deps.vision.release(handle)
      this.emitProgress()
    }
  }

  private async processSourcePage(
    pageIndex: number,
    handle: PreparedPageHandle,
    seamBandPx: number,
    onEdgeText: (edgeText: SourceEdgeText) => void,
  ): Promise<void> {
    try {
      const { overlays } = await this.processPreparedPage(handle, false, onEdgeText, seamBandPx)
      for (const overlay of overlays) {
        this.emitMergedOverlay(overlay)
      }

      this.emitStatus(pageIndex, 'done')
    } catch (error) {
      if (this.abort.signal.aborted || this.finished) return
      this.markPageError(pageIndex, error)
    } finally {
      this.emitProgress()
    }
  }

  private async processSeamRepair(
    top: PreparedPageHandle,
    bottom: PreparedPageHandle,
    bandPx: number,
  ): Promise<void> {
    const createSeamRepair = this.deps.vision.createSeamRepair
    if (!createSeamRepair) return

    let seam: PreparedPageHandle | null = null
    try {
      seam = await createSeamRepair.call(this.deps.vision, top, bottom, bandPx, this.abort.signal)
      if (!seam || this.abort.signal.aborted || this.finished) return

      const { overlays } = await this.processPreparedPage(seam, true)
      for (const overlay of overlays) {
        this.emitMergedOverlay(overlay)
      }
    } finally {
      if (seam) this.deps.vision.release(seam)
    }
  }

  private async processPreparedPage(
    handle: PreparedPageHandle,
    repairOnly: boolean,
    onEdgeText?: (edgeText: SourceEdgeText) => void,
    seamBandPx = DEFAULT_SEAM_BAND_PX,
  ): Promise<PreparedPageProcessResult> {
    const signal = this.abort.signal

    const detectTask = this.deps.executor.detect(async () => {
      if (!repairOnly) this.emitProjectionStatuses(handle, 'detecting')
      return this.deps.vision.detectTextRegions(handle, signal)
    }).catch(() => null)

    if (!repairOnly) this.emitProjectionStatuses(handle, 'ocr')
    const text = await this.deps.executor.ocr(async () => {
      const options = {
        pageIndex: handle.preparedPageIndex,
        sourceLang: this.request.sourceLanguage,
        signal,
      }
      const encoded = this.deps.recognizer.recognizeEncoded
        ? await this.deps.vision.encodeForOcr(handle, signal)
        : null
      const recognized = encoded
        ? await this.deps.recognizer.recognizeEncoded!(encoded, options)
        : await this.deps.recognizer.recognizeText(
            await this.deps.vision.readPixels(handle, signal),
            options,
          )
      if (!repairOnly) onEdgeText?.(edgeTextFromRecognition(recognized, seamBandPx))
      return preparedTextFromRecognition({ handle, recognized })
    })

    if (!repairOnly) this.emitProjectionStatuses(handle, 'translating')
    const preparedResult = await this.deps.executor.translate(() =>
      translatePreparedText({
        text,
        translator: this.deps.translator,
        sourceLanguage: this.request.sourceLanguage,
        targetLanguage: this.request.targetLanguage,
        signal,
      }),
    )

    if (!repairOnly) this.emitProjectionStatuses(handle, 'composing')
    const regions = await detectTask
    if (signal.aborted || this.finished) return { overlays: [] }

    const placements = buildOverlayPlacements({
      recognized: preparedResult.recognized,
      textUnits: preparedResult.textUnits,
      regions,
    })
    const placementMargins = await this.deps.vision.estimateMargins(handle, placements, signal)

    return {
      overlays: composeAndProjectOverlays({
        handle,
        recognized: preparedResult.recognized,
        textUnits: preparedResult.textUnits,
        translations: preparedResult.translations,
        regions,
        placements,
        placementMargins,
      }),
    }
  }

  private emitProjectionStatuses(handle: PreparedPageHandle, status: PageTranslationStatus): void {
    for (const proj of handle.projections) this.emitStatus(proj.sourcePageIndex, status)
  }

  private emitStatus(pageIndex: number, status: PageTranslationStatus): void {
    if (isTerminalPageStatus(this.pageStatuses.get(pageIndex))) return
    this.pageStatuses.set(pageIndex, status)
    this.emit({ type: 'page-status', pageIndex, status })
  }

  private emitMergedOverlay(incoming: PageOverlay): void {
    const existingIndex = this.overlays.findIndex(overlay => overlay.pageIndex === incoming.pageIndex)
    const merged = mergePageOverlay(existingIndex === -1 ? undefined : this.overlays[existingIndex], incoming)
    if (existingIndex === -1) this.overlays.push(merged)
    else this.overlays[existingIndex] = merged
    this.emit({ type: 'page-overlay', overlay: merged })
  }

  private markPageError(pageIndex: number, error: unknown): void {
    if (this.pageStatuses.get(pageIndex) === 'done') return
    const err = error instanceof Error ? error : new Error(String(error))
    this.pageStatuses.set(pageIndex, 'error')
    this.emit({ type: 'page-status', pageIndex, status: 'error', error: err })
    this.emitProgress()
  }

  private setInitialPageStatuses(pageIndexes: readonly number[]): void {
    this.pageStatuses.clear()
    for (const pageIndex of pageIndexes) {
      this.pageStatuses.set(pageIndex, 'queued')
      this.emit({ type: 'page-status', pageIndex, status: 'queued' })
    }
  }

  private emitProgress(): void {
    let done = 0
    let total = 0
    for (const status of this.pageStatuses.values()) {
      total++
      if (status === 'done' || status === 'error') done++
    }
    if (this.lastProgress?.done === done && this.lastProgress.total === total) return
    this.lastProgress = { done, total }
    this.emit({ type: 'progress', progress: this.lastProgress })
  }

  private emit(event: TranslationRunEvent): void {
    if (this.finished && !isTerminalEvent(event)) return
    for (const listener of this.listeners) {
      try {
        listener(event)
      } catch {
        // Listener error must not kill the run.
      }
    }
  }

  private finishCompleted(): void {
    if (this.finished) return
    this.finished = true
    this.emit({ type: 'completed', overlays: this.overlays })
    this.doneResolve(this.overlays)
  }

  private finishCancelled(): void {
    if (this.finished) return
    this.finished = true
    this.emit({ type: 'cancelled' })
    this.doneResolve(this.overlays)
  }

  private finishFailed(error: Error): void {
    if (this.finished) return
    this.finished = true
    this.emit({ type: 'failed', error })
    this.doneReject(error)
  }
}

function isTerminalEvent(event: TranslationRunEvent): boolean {
  return event.type === 'completed' || event.type === 'failed' || event.type === 'cancelled'
}

function isTerminalPageStatus(status: PageTranslationStatus | undefined): boolean {
  return status === 'done' || status === 'error'
}

function sourcePageIndexes(handle: PreparedPageHandle): number[] {
  return [...new Set(handle.projections.map(projection => projection.sourcePageIndex))]
}

function mergePageOverlay(existing: PageOverlay | undefined, incoming: PageOverlay): PageOverlay {
  const translations = existing
    ? mergeTranslations(existing.translations, incoming.translations)
    : incoming.translations
  const byUnitId = new Map(translations.map(unit => [unit.unitId, unit]))
  const mergedItems = dedupePlacementItems([
    ...(existing ? placementItems(existing) : []),
    ...placementItems(incoming),
  ], byUnitId)
  const stitchedItems = stitchContinuationPlacementItems(mergedItems, byUnitId)

  return {
    pageIndex: existing?.pageIndex ?? incoming.pageIndex,
    pageSize: existing?.pageSize ?? incoming.pageSize,
    placements: stitchedItems.map(item => item.placement),
    translations,
    placementMargins: stitchedItems.map(item => item.margin),
  }
}

function placementItems(overlay: PageOverlay): OverlayPlacementItem[] {
  return overlay.placements.map((placement, index) => ({ placement, margin: overlay.placementMargins[index]! }))
}

function dedupePlacementItems(
  items: readonly OverlayPlacementItem[],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): OverlayPlacementItem[] {
  return items
    .map((item, index) => ({ ...item, index, score: placementScore(item.placement, translations) }))
    .sort((a, b) => b.score - a.score)
    .reduce<Array<{
      readonly placement: PageOverlay['placements'][number]
      readonly margin: PageOverlay['placementMargins'][number]
      readonly index: number
    }>>((kept, item) => {
      if (!kept.some(candidate => duplicatePlacement(candidate.placement, item.placement, translations))) kept.push(item)
      return kept
    }, [])
    .sort((a, b) => a.index - b.index)
    .map(({ placement, margin }) => ({ placement, margin }))
}

function stitchContinuationPlacementItems(
  items: readonly OverlayPlacementItem[],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): OverlayPlacementItem[] {
  const out = items.map((item, index) => ({ ...item, index }))

  for (;;) {
    const pair = bestContinuationPair(out, translations)
    if (!pair) break

    const [aIndex, bIndex] = pair
    const a = out[aIndex]!
    const b = out[bIndex]!
    const merged = mergeContinuationItems(a, b)
    out.splice(Math.max(aIndex, bIndex), 1)
    out.splice(Math.min(aIndex, bIndex), 1)
    out.push({ ...merged, index: Math.min(a.index, b.index) })
  }

  return out
    .sort((a, b) => a.index - b.index)
    .map(({ placement, margin }) => ({ placement, margin }))
}

function bestContinuationPair(
  items: readonly (OverlayPlacementItem & { readonly index: number })[],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): readonly [number, number] | null {
  let best: { readonly pair: readonly [number, number]; readonly score: number } | null = null
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const score = continuationScore(items[i]!, items[j]!, translations)
      if (score === null) continue
      if (!best || score < best.score) best = { pair: [i, j], score }
    }
  }
  return best?.pair ?? null
}

function continuationScore(
  a: OverlayPlacementItem,
  b: OverlayPlacementItem,
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): number | null {
  if (a.placement.pageIndex !== b.placement.pageIndex) return null
  if (a.placement.role !== b.placement.role) return null
  if (a.placement.role !== 'dialogue' && a.placement.role !== 'narration') return null
  if (isVerticalPlacement(a.placement) || isVerticalPlacement(b.placement)) return null
  if (!hasRenderableUnits(a.placement, translations) || !hasRenderableUnits(b.placement, translations)) return null

  const [top, bottom] = sortVerticalItems(a, b)
  const sourceTop = placementSourceText(top.placement, translations)
  const sourceBottom = placementSourceText(bottom.placement, translations)
  if (!isLikelyContinuationText(sourceTop, sourceBottom)) return null

  const topBox = placementVisualBBox(top.placement)
  const bottomBox = placementVisualBBox(bottom.placement)
  const fontPx = localPlacementFontPx(top.placement, bottom.placement)
  const gap = yGap(topBox, bottomBox)
  if (gap > fontPx * 0.65) return null

  const overlap = xOverlap(topBox, bottomBox)
  const bottomCenterInsideTop = centerXOf(bottomBox) >= topBox[0] - fontPx * 0.5
    && centerXOf(bottomBox) <= topBox[2] + fontPx * 0.5
  if (overlap < 0.55 && !bottomCenterInsideTop) return null
  if (!compatibleSafeBounds(top.margin, bottom.margin, fontPx)) return null

  return gap / fontPx + (1 - overlap)
}

function sortVerticalItems<T extends OverlayPlacementItem>(a: T, b: T): readonly [T, T] {
  return placementVisualBBox(a.placement)[1] <= placementVisualBBox(b.placement)[1]
    ? [a, b]
    : [b, a]
}

function isVerticalPlacement(placement: PageOverlay['placements'][number]): boolean {
  return placement.layoutHint.direction === 'vertical' || placement.fontHint?.sourceDirection === 'vertical'
}

function hasRenderableUnits(
  placement: PageOverlay['placements'][number],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): boolean {
  return placement.sourceUnitIds.some(id => {
    const unit = translations.get(id)
    return !!unit && unit.kind !== 'skip' && !!unit.targetText.trim()
  })
}

function isLikelyContinuationText(top: string, bottom: string): boolean {
  if (containsCreditMarker(top) || containsCreditMarker(bottom)) return false
  const a = normalizeText(top)
  const b = normalizeText(bottom)
  if (!a || !b) return false
  if (a.includes(b) || b.includes(a)) return false
  return !/[。.!！?？…」』）)]$/u.test(top.trim()) || /^[和及与與并並而或的了吗嗎呢吧啊呀]/u.test(bottom.trim())
}

function containsCreditMarker(text: string): boolean {
  return /https?|www|\.com|baozimh|包子漫[画畫]|騰訊|腾讯|tencent/iu.test(text)
}

function compatibleSafeBounds(
  a: PageOverlay['placementMargins'][number],
  b: PageOverlay['placementMargins'][number],
  fontPx: number,
): boolean {
  if (!compatibleBackground(a, b)) return false
  if (yGap(a.safeBounds, b.safeBounds) > fontPx * 0.8) return false
  return xOverlap(a.safeBounds, b.safeBounds) >= 0.25 || centerDistanceRatio(a.safeBounds, b.safeBounds) <= 0.45
}

function compatibleBackground(
  a: PageOverlay['placementMargins'][number],
  b: PageOverlay['placementMargins'][number],
): boolean {
  if (!a.backgroundRgb || !b.backgroundRgb) return true
  const distance = Math.hypot(
    a.backgroundRgb[0] - b.backgroundRgb[0],
    a.backgroundRgb[1] - b.backgroundRgb[1],
    a.backgroundRgb[2] - b.backgroundRgb[2],
  )
  return distance <= 48
}

function localPlacementFontPx(
  a: PageOverlay['placements'][number],
  b: PageOverlay['placements'][number],
): number {
  return median([
    a.fontHint?.sourceFontPx ?? 0,
    b.fontHint?.sourceFontPx ?? 0,
    Math.min(bboxWidth(placementVisualBBox(a)), bboxHeight(placementVisualBBox(a))),
    Math.min(bboxWidth(placementVisualBBox(b)), bboxHeight(placementVisualBBox(b))),
  ].filter(value => value > 0)) || 24
}

function mergeContinuationItems(
  a: OverlayPlacementItem,
  b: OverlayPlacementItem,
): OverlayPlacementItem {
  const [top, bottom] = sortVerticalItems(a, b)
  const bbox = unionBBoxes([top.placement.bbox, bottom.placement.bbox])
  const pageSize: readonly [number, number] = [top.placement.pageSize[0], top.placement.pageSize[1]]
  const mergedPlacement: PageOverlay['placements'][number] = {
    ...top.placement,
    id: `${top.placement.id}+${bottom.placement.id}`,
    sourceUnitIds: [...top.placement.sourceUnitIds, ...bottom.placement.sourceUnitIds],
    drawable: bboxToPolygon(bbox),
    bbox,
    textBoxes: [...top.placement.textBoxes, ...bottom.placement.textBoxes],
    rotationDeg: Math.abs(top.placement.rotationDeg) >= Math.abs(bottom.placement.rotationDeg) ? top.placement.rotationDeg : bottom.placement.rotationDeg,
    confidence: Math.max(top.placement.confidence, bottom.placement.confidence),
    fontHint: mergePlacementFontHints(top.placement, bottom.placement),
  }
  return {
    placement: mergedPlacement,
    margin: mergePlacementMargins(top.margin, bottom.margin, bbox, pageSize),
  }
}

function mergePlacementFontHints(
  a: PageOverlay['placements'][number],
  b: PageOverlay['placements'][number],
): PageOverlay['placements'][number]['fontHint'] {
  type PlacementFontHint = NonNullable<PageOverlay['placements'][number]['fontHint']>
  const hints = [a.fontHint, b.fontHint].filter((hint): hint is PlacementFontHint => !!hint)
  if (!hints.length) return null
  const sourceFontPx = Math.round(median(hints.map(hint => hint.sourceFontPx ?? 0).filter(value => value > 0)))
  const sourceLineCount = hints.reduce((sum, hint) => sum + (hint.sourceLineCount ?? 1), 0)
  const charsPerLine = hints.reduce((sum, hint) => sum + (hint.sourceAvgCharsPerLine ?? 0) * (hint.sourceLineCount ?? 1), 0) / Math.max(1, sourceLineCount)
  return {
    sourceFontPx: sourceFontPx || undefined,
    sourceLineCount,
    sourceAvgCharsPerLine: charsPerLine,
    sourceDirection: 'horizontal',
  }
}

function mergePlacementMargins(
  a: PageOverlay['placementMargins'][number],
  b: PageOverlay['placementMargins'][number],
  bbox: BBox,
  pageSize: readonly [number, number],
): PageOverlay['placementMargins'][number] {
  const safeBounds = clipBBox(unionBBoxes([a.safeBounds, b.safeBounds, bbox]), pageSize)
  const componentBBox = a.componentBBox || b.componentBBox
    ? clipBBox(unionBBoxes([...(a.componentBBox ? [a.componentBBox] : []), ...(b.componentBBox ? [b.componentBBox] : [])]), pageSize)
    : null
  return {
    reasons: { top: 'stitched', right: 'stitched', bottom: 'stitched', left: 'stitched', overall: 'stitched' },
    margins: marginsWithinBounds(bbox, safeBounds),
    backgroundRgb: a.backgroundRgb ?? b.backgroundRgb,
    backgroundTolerance: Math.max(a.backgroundTolerance, b.backgroundTolerance),
    safeBounds,
    componentBBox,
    componentConfidence: Math.min(a.componentConfidence || 1, b.componentConfidence || 1),
    shape: null,
  }
}

function mergeTranslations(
  first: PageOverlay['translations'],
  second: PageOverlay['translations'],
): PageOverlay['translations'] {
  const out = new Map<string, PageOverlay['translations'][number]>()
  for (const unit of first) out.set(unit.unitId, unit)
  for (const unit of second) {
    const existing = out.get(unit.unitId)
    out.set(unit.unitId, existing ? betterTranslation(existing, unit) : unit)
  }
  return [...out.values()]
}

function betterTranslation(
  a: PageOverlay['translations'][number],
  b: PageOverlay['translations'][number],
): PageOverlay['translations'][number] {
  return translationScore(b) > translationScore(a) ? b : a
}

function translationScore(unit: PageOverlay['translations'][number]): number {
  return (unit.kind === 'skip' ? 0 : 1000)
    + normalizeText(unit.targetText).length * 2
    + normalizeText(unit.sourceText).length
}

function duplicatePlacement(
  a: PageOverlay['placements'][number],
  b: PageOverlay['placements'][number],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): boolean {
  const sourceA = placementSourceText(a, translations)
  const sourceB = placementSourceText(b, translations)
  const sourceContained = containedText(sourceA, sourceB)
  const overlap = placementOverlapRatio(a, b)
  if (sourceContained && overlap >= 0.65) return true
  if (!compatiblePlacementRole(a.role, b.role)) return false
  if (samePlacementGeometry(a, b)) return true
  const textRelated = relatedText(placementText(a, translations), placementText(b, translations))
  const sourceSimilar = similarText(sourceA, sourceB)
  if (overlap >= 0.65) return textRelated || sourceSimilar || sourceContained
  if (overlap >= 0.35 && sourceContained) return true
  if (overlap >= 0.55 && sourceSimilar) return true
  if (overlap >= 0.35 && textRelated) return true
  if ((sourceSimilar || sourceContained) && visualCenterDistanceRatio(a, b) <= 0.30) return true
  return textRelated && centerDistanceRatio(a.bbox, b.bbox) <= 0.35
}

function samePlacementGeometry(
  a: PageOverlay['placements'][number],
  b: PageOverlay['placements'][number],
): boolean {
  return sameGeometryBox(a.bbox, b.bbox) || sameGeometryBox(placementVisualBBox(a), placementVisualBBox(b))
}

function placementOverlapRatio(
  a: PageOverlay['placements'][number],
  b: PageOverlay['placements'][number],
): number {
  return Math.max(
    bboxOverlapRatio(a.bbox, b.bbox),
    bboxOverlapRatio(placementVisualBBox(a), placementVisualBBox(b)),
  )
}

function placementVisualBBox(placement: PageOverlay['placements'][number]): BBox {
  return rectBBox(textFitRect(placement))
}

function visualCenterDistanceRatio(
  a: PageOverlay['placements'][number],
  b: PageOverlay['placements'][number],
): number {
  return centerDistanceRatio(placementVisualBBox(a), placementVisualBBox(b))
}

function compatiblePlacementRole(a: PageOverlay['placements'][number]['role'], b: PageOverlay['placements'][number]['role']): boolean {
  if (a === b) return true
  return a !== 'sfx' && b !== 'sfx'
}

function placementScore(
  placement: PageOverlay['placements'][number],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): number {
  const text = placementText(placement, translations)
  return normalizeText(text).length * 10
    + placement.sourceUnitIds.length * 6
    + placement.textBoxes.length * 2
    + bboxArea(placement.bbox) / 10_000
}

function placementText(
  placement: PageOverlay['placements'][number],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): string {
  return placement.sourceUnitIds
    .map(id => translations.get(id))
    .filter((unit): unit is PageOverlay['translations'][number] => !!unit)
    .map(unit => `${unit.sourceText}\n${unit.targetText}`)
    .join('\n')
}

function placementSourceText(
  placement: PageOverlay['placements'][number],
  translations: ReadonlyMap<string, PageOverlay['translations'][number]>,
): string {
  return placement.sourceUnitIds
    .map(id => translations.get(id))
    .filter((unit): unit is PageOverlay['translations'][number] => !!unit)
    .map(unit => unit.sourceText)
    .join('\n')
}

function relatedText(a: string, b: string): boolean {
  const left = normalizeText(a)
  const right = normalizeText(b)
  if (!left || !right) return false
  return left.includes(right) || right.includes(left)
}

function similarText(a: string, b: string): boolean {
  const left = normalizeText(a)
  const right = normalizeText(b)
  if (left.length < 4 || right.length < 4) return false
  if (left.includes(right) || right.includes(left)) return true
  const common = longestCommonSubstringLength(left, right)
  return common / Math.min(left.length, right.length) >= 0.45
}

function containedText(a: string, b: string): boolean {
  const left = normalizeText(a)
  const right = normalizeText(b)
  if (!left || !right) return false
  return left.includes(right) || right.includes(left)
}

function longestCommonSubstringLength(a: string, b: string): number {
  let previous = new Array(b.length + 1).fill(0) as number[]
  let best = 0
  for (let i = 1; i <= a.length; i++) {
    const current = new Array(b.length + 1).fill(0) as number[]
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] !== b[j - 1]) continue
      const length = previous[j - 1]! + 1
      current[j] = length
      if (length > best) best = length
    }
    previous = current
  }
  return best
}

function normalizeText(value: string): string {
  return value.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
}

function xOverlap(a: BBox, b: BBox): number {
  const overlap = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  return overlap / Math.max(1, Math.min(bboxWidth(a), bboxWidth(b)))
}

function yGap(a: BBox, b: BBox): number {
  return Math.max(0, Math.max(a[1], b[1]) - Math.min(a[3], b[3]))
}

function centerXOf(bbox: BBox): number {
  return (bbox[0] + bbox[2]) / 2
}

function unionBBoxes(boxes: readonly BBox[]): BBox {
  return [
    Math.min(...boxes.map(box => box[0])),
    Math.min(...boxes.map(box => box[1])),
    Math.max(...boxes.map(box => box[2])),
    Math.max(...boxes.map(box => box[3])),
  ]
}

function bboxToPolygon(bbox: BBox): PageOverlay['placements'][number]['drawable'] {
  return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
}

function clipBBox(bbox: BBox, pageSize: readonly [number, number]): BBox {
  return [Math.max(0, bbox[0]), Math.max(0, bbox[1]), Math.min(pageSize[0], bbox[2]), Math.min(pageSize[1], bbox[3])]
}

function marginsWithinBounds(bbox: BBox, bounds: BBox): PageOverlay['placementMargins'][number]['margins'] {
  return {
    top: Math.max(0, bbox[1] - bounds[1]),
    right: Math.max(0, bounds[2] - bbox[2]),
    bottom: Math.max(0, bounds[3] - bbox[3]),
    left: Math.max(0, bbox[0] - bounds[0]),
  }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}

function bboxOverlapRatio(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  const intersection = (ix2 - ix1) * (iy2 - iy1)
  return intersection / Math.max(1, Math.min(bboxArea(a), bboxArea(b)))
}

function sameGeometryBox(a: BBox, b: BBox): boolean {
  const maxWidth = Math.max(1, a[2] - a[0], b[2] - b[0])
  const maxHeight = Math.max(1, a[3] - a[1], b[3] - b[1])
  const acx = (a[0] + a[2]) / 2
  const acy = (a[1] + a[3]) / 2
  const bcx = (b[0] + b[2]) / 2
  const bcy = (b[1] + b[3]) / 2
  const centerDistance = Math.hypot(acx - bcx, acy - bcy)
  const diagonal = Math.hypot(maxWidth, maxHeight)
  const sizeDelta = Math.max(
    Math.abs((a[2] - a[0]) - (b[2] - b[0])) / maxWidth,
    Math.abs((a[3] - a[1]) - (b[3] - b[1])) / maxHeight,
  )
  return centerDistance / Math.max(1, diagonal) <= OVERLAY_GEOMETRY_TWIN_NOISE
    && sizeDelta <= OVERLAY_GEOMETRY_TWIN_NOISE
    && bboxOverlapRatio(a, b) >= 1 - OVERLAY_GEOMETRY_TWIN_NOISE
}

function centerDistanceRatio(a: BBox, b: BBox): number {
  const acx = (a[0] + a[2]) / 2
  const acy = (a[1] + a[3]) / 2
  const bcx = (b[0] + b[2]) / 2
  const bcy = (b[1] + b[3]) / 2
  const distance = Math.hypot(acx - bcx, acy - bcy)
  const scale = Math.max(
    1,
    Math.max(a[2] - a[0], a[3] - a[1]),
    Math.max(b[2] - b[0], b[3] - b[1]),
  )
  return distance / scale
}

function bboxArea(bbox: BBox): number {
  return bboxWidth(bbox) * bboxHeight(bbox)
}

function bboxWidth(bbox: BBox): number {
  return Math.max(0, bbox[2] - bbox[0])
}

function bboxHeight(bbox: BBox): number {
  return Math.max(0, bbox[3] - bbox[1])
}

function edgeTextFromRecognition(recognized: RecognizedTextPage, seamBandPx: number): SourceEdgeText {
  const [, pageHeight] = recognized.pageSize
  const bandPx = edgeTextBandPx(pageHeight, seamBandPx)
  const samples = recognized.blocks.flatMap(edgeTextSamples)
  const bodyGlyphPx = edgeBodyGlyphPx(samples)
  let top = false
  let bottom = false

  for (const sample of samples) {
    if (!isReadableEdgeText(sample, recognized.pageSize, bodyGlyphPx)) continue
    if (sample.bbox[1] <= bandPx) top = true
    if (sample.bbox[3] >= pageHeight - bandPx) bottom = true
    if (top && bottom) return { top, bottom }
  }

  return { top, bottom }
}

function edgeTextSamples(block: TextBlock): readonly EdgeTextSample[] {
  if (isLikelyPageMarker(block.text)) return []
  const markerLines = block.lines.filter(line => isLikelyPageMarker(line.text))
  if (block.words.length) return block.words
    .filter(word => !markerLines.some(line => containsBBoxCenter(line.bbox, word.bbox)))
    .map(word => ({ bbox: word.bbox, text: word.text }))
  if (block.lines.length) return block.lines
    .filter(line => !markerLines.includes(line))
    .map(line => ({ bbox: line.bbox, text: line.text }))
  return [{ bbox: block.bbox, text: block.text }]
}

function containsBBoxCenter(outer: BBox, inner: BBox): boolean {
  const cx = (inner[0] + inner[2]) / 2
  const cy = (inner[1] + inner[3]) / 2
  return outer[0] <= cx && cx <= outer[2] && outer[1] <= cy && cy <= outer[3]
}

function edgeTextBandPx(pageHeight: number, seamBandPx: number): number {
  return Math.max(1, Math.min(pageHeight, Math.round(seamBandPx)))
}

function isReadableEdgeText(sample: EdgeTextSample, pageSize: readonly [number, number], bodyGlyphPx: number): boolean {
  if (!hasTranslatableLetters(sample.text)) return false
  const bbox = sample.bbox
  if (!isInsideEdgeTextWindow(bbox, pageSize)) return false
  const glyphPx = Math.min(Math.max(0, bbox[2] - bbox[0]), Math.max(0, bbox[3] - bbox[1]))
  const minGlyphPx = Math.max(
    EDGE_TEXT_MIN_GLYPH_PX,
    Math.round(Math.min(...pageSize) * EDGE_TEXT_MIN_GLYPH_RATIO),
    Math.round(bodyGlyphPx * 0.45),
  )
  return glyphPx >= minGlyphPx
}

function isInsideEdgeTextWindow(bbox: BBox, pageSize: readonly [number, number]): boolean {
  const [pageWidth] = pageSize
  const windowLeft = pageWidth * EDGE_TEXT_WINDOW_LEFT_RATIO
  const windowRight = pageWidth * EDGE_TEXT_WINDOW_RIGHT_RATIO
  const centerX = (bbox[0] + bbox[2]) / 2
  const bboxWidth = Math.max(1, bbox[2] - bbox[0])
  const overlap = Math.max(0, Math.min(bbox[2], windowRight) - Math.max(bbox[0], windowLeft))
  return centerX >= windowLeft && centerX <= windowRight && overlap / bboxWidth >= 0.8
}

function edgeBodyGlyphPx(samples: readonly EdgeTextSample[]): number {
  const glyphs = samples
    .filter(sample => hasTranslatableLetters(sample.text))
    .map(sample => Math.min(
      Math.max(0, sample.bbox[2] - sample.bbox[0]),
      Math.max(0, sample.bbox[3] - sample.bbox[1]),
    ))
    .filter(glyph => glyph > 0)
    .sort((a, b) => a - b)
  if (!glyphs.length) return 0
  return glyphs[Math.floor((glyphs.length - 1) * 0.7)] ?? 0
}

function hasTranslatableLetters(text: string): boolean {
  const value = text.trim()
  if (!value || isLikelyPageMarker(value) || isLikelyPageMarkerToken(value)) return false
  return /\p{L}/u.test(value)
}

function isLikelyPageMarker(text: string): boolean {
  const compact = compactPageMarkerText(text)
  if (!compact) return false
  if (/^(?:p|pg|page)[\d０-９]+(?:of[\d０-９]+)?$/iu.test(compact)) return true
  if (/^第[\d０-９一二三四五六七八九十百千]+[页頁]$/u.test(compact)) return true
  if (/^[\d０-９]+[页頁]$/u.test(compact)) return true
  return /^[IVXLCDM]{1,8}$/i.test(compact)
}

function isLikelyPageMarkerToken(text: string): boolean {
  const compact = compactPageMarkerText(text)
  return /^(?:p|pg|page)$/iu.test(compact) || /^[第页頁]$/u.test(compact)
}

function compactPageMarkerText(text: string): string {
  return text.replace(/[^\p{L}\p{N}一二三四五六七八九十百千]+/gu, '')
}
