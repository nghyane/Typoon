// reader/translation.ts — ReaderTranslation controller (public API).
//
// Thin orchestrator composed of focused modules:
//   - TranslationStore-like state held here, emitted with stable shape
//   - PageScheduler    : stable page identity + viewport-aware ordering
//   - PagePipeline     : pure capture → OCR → detect → translate
//   - OverlayManager   : attach/detach + working hide
//   - PageProvider     : page loading + LRU memory bound
//   - visionRuntime    : model/ORT singleton + model state
//
// Public state shape is intentionally preserved for UI compatibility.

import { detectBrowserCapabilities } from '../adapters/browserCapabilities'
import type { ChapterContentLayout } from '../domain/chapterContent'
import type { PageScanUnit, ReaderPageOverlay } from '../domain/pageScan'
import { ensureMangaFontLoaded } from '../render/font'
import { LensTextRecognizer } from '../recognizers/lens/LensTextRecognizer'
import type { SourcePageSize } from '../pipeline/chapterContent'
import { DeepLTranslateWeb } from '../translators/deepl-web/DeepLTranslateWeb'
import { GoogleTranslateWeb } from '../translators/google-web/GoogleTranslateWeb'
import type { Translator } from '../translators/translator'
import { PagePipeline, deduplicateSeamBlocks } from './pagePipeline'
import { PageScheduler } from './pageScheduler'
import { planPageScans, sourceUsesHalo, measuredPagesFromLayout, type MeasuredPage } from './pageScan'
import { measureLayout, visibleContentRange } from './chunkCapture'
import { OverlayManager } from './overlayManager'
import type { ReaderRenderer } from './renderer'
import { PageProvider, type LoadedPage } from './pageProvider'
import { defaultTranslationConfig, type TranslationConfig } from './translationConfig'
import { errorMessage, throwIfAborted, yieldToIdle } from './asyncSignal'
import {
  prewarmTextRegionDetector,
  subscribeModelState,
  type ReaderModelState,
} from './visionRuntime'

export type ReaderPhase = 'idle' | 'loading' | 'ready' | 'translating' | 'done' | 'error'

/** Translation backend the reader uses. AI/custom LLM gateway is future work. */
export type TranslationProvider = 'deepl' | 'google'

export interface ReaderTranslationState {
  phase: ReaderPhase
  prepare: { done: number; total: number; preparedPages: number }
  translate: { done: number; total: number }
  model: ReaderModelState
  sourceLanguage: string | null
  targetLanguage: string
  hidden: boolean
  /** Chunks skipped after exhausting retries (partial translation). */
  failed: number
  error?: string
}

export type { ReaderModelState }

type Listener = (state: ReaderTranslationState) => void

export interface ReaderTranslationChapter {
  readonly chapterKey: string
  readonly pageCount: number
  readonly readPage: (index: number, signal?: AbortSignal) => Promise<Blob>
  readonly sourceLanguage: string | null
  readonly targetLanguage: string
  /**
   * Authoritative source size per page, when the host already decoded it (e.g.
   * the reader's page cache). Lets the provider skip a redundant decode and, more
   * importantly, makes `unit.source` identical to the size the host renders the
   * page frame with — so overlay geometry and the displayed image never diverge.
   */
  readonly pageSize?: (index: number) => SourcePageSize | null
}

export interface ReaderTranslationOptions {
  readonly config?: TranslationConfig
  readonly provider?: TranslationProvider
  /** Inject the render surface. Defaults to the imperative DOM OverlayManager. */
  readonly renderer?: ReaderRenderer
}

function init(pageCount: number, sourceLang: string | null, targetLang: string, model: ReaderModelState): ReaderTranslationState {
  return {
    phase: 'idle',
    prepare: { done: 0, total: pageCount, preparedPages: 0 },
    translate: { done: 0, total: pageCount },
    model,
    sourceLanguage: sourceLang,
    targetLanguage: targetLang,
    hidden: false,
    failed: 0,
  }
}

export class ReaderTranslation {
  private readonly listeners = new Set<Listener>()
  private readonly recognizer = new LensTextRecognizer()
  private readonly overlays: ReaderRenderer
  private readonly scheduler = new PageScheduler()
  private readonly config: TranslationConfig
  private readonly pipeline: PagePipeline

  private translator: Translator | null = null
  private translatorProvider: TranslationProvider | null = null
  private provider: TranslationProvider
  private chapter: ReaderTranslationChapter | null = null
  private pages: PageProvider | null = null
  private state: ReaderTranslationState
  private pageOverlays = new Map<number, ReaderPageOverlay>()
  private units: readonly PageScanUnit[] = []
  private abort: AbortController | null = null
  private generation = 0
  private overlayRevision = 0
  private active = false
  private draining = false
  private latestModel: ReaderModelState = { state: 'idle' }
  private readonly unsubscribeModelState: () => void

  constructor(options: ReaderTranslationOptions = {}) {
    this.config = options.config ?? defaultTranslationConfig
    this.provider = options.provider ?? 'deepl'
    this.overlays = options.renderer ?? new OverlayManager(this.config.chunk.overlayMarginPx)
    this.pipeline = new PagePipeline({
      recognizer: this.recognizer,
      translator: () => this.ensureTranslator(),
      config: this.config,
    })
    this.state = init(0, null, 'vi', this.latestModel)
    this.unsubscribeModelState = subscribeModelState(model => {
      this.latestModel = model
      this.setState({ model })
    })
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn)
    fn(this.state)
    return () => this.listeners.delete(fn)
  }

  registerContentHost(host: HTMLElement): () => void {
    this.overlays.setHost(host)
    return () => {
      if (this.overlays.currentHost === host) {
        this.overlays.detach()
        this.overlays.setHost(null)
      }
    }
  }

  /** Register a page element (called from a Svelte action). Returns cleanup. */
  registerPage(pageIndex: number, el: HTMLElement): () => void {
    return this.overlays.registerPage(pageIndex, el)
  }

  setChapter(chapter: ReaderTranslationChapter): void {
    if (this.isSameChapter(chapter)) return
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    this.chapter = chapter
    this.active = chapter.pageCount > 0
    this.state = {
      ...init(chapter.pageCount, chapter.sourceLanguage, chapter.targetLanguage, this.latestModel),
      phase: this.active ? 'ready' : 'idle',
    }
    this.emit()
    if (this.active) {
      void ensureMangaFontLoaded()
      prewarmTextRegionDetector(this.config)
    }
  }

  clear(): void {
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    this.chapter = null
    this.active = false
    this.state = init(0, this.state.sourceLanguage, this.state.targetLanguage, this.latestModel)
    this.emit()
  }

  translate(): void {
    if (!this.active || !this.chapter?.pageCount) return
    if (this.abort && !this.abort.signal.aborted) return
    const generation = ++this.generation
    void this.start(generation)
  }

  /** Toggle translation visibility. Fixes the broken hide behavior. */
  setHidden(hidden: boolean): void {
    this.overlays.setHidden(hidden)
    this.setState({ hidden })
  }

  /** Choose the translation backend. Applies to the next translate() run. */
  setProvider(provider: TranslationProvider): void {
    this.provider = provider
  }

  cancel(): void {
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    if (this.chapter?.pageCount) {
      this.active = true
      this.state = { ...init(this.chapter.pageCount, this.chapter.sourceLanguage, this.chapter.targetLanguage, this.latestModel), phase: 'ready' }
    } else {
      this.active = false
      this.state = init(0, this.state.sourceLanguage, this.state.targetLanguage, this.latestModel)
    }
    this.emit()
  }

  dispose(): void {
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    this.chapter = null
    this.active = false
    this.listeners.clear()
    this.overlays.dispose()
    this.unsubscribeModelState()
    this.disposeTranslator()
  }

  private async start(generation: number): Promise<void> {
    const chapter = this.chapter
    if (!chapter) return

    this.stopRun()
    const abort = new AbortController()
    this.abort = abort

    this.setState({
      phase: 'loading',
      prepare: { done: 0, total: chapter.pageCount, preparedPages: 0 },
      translate: { done: 0, total: chapter.pageCount },
      failed: 0,
      error: undefined,
    })

    try {
      this.pages = new PageProvider({
        pageCount: chapter.pageCount,
        maxCachedPages: this.config.memory.maxCachedPages,
        readPage: (index, signal) => chapter.readPage(index, signal),
        pageSize: chapter.pageSize,
        onProgress: loaded => this.setState({
          prepare: { done: loaded, total: chapter.pageCount, preparedPages: loaded },
        }),
      })

      await yieldAfterPaint()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      this.clearOverlayKeepActive()
      await yieldToBrowser()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      await ensureMangaFontLoaded()
      if (!this.isCurrent(generation) || abort.signal.aborted) return

      const layout = this.measure()
      if (!layout) throw new Error('chapter content host is not ready')
      this.rebuildPlan(layout)
      this.pageOverlays = new Map()

      this.setState({ phase: 'translating', translate: { done: 0, total: chapter.pageCount } })
      await yieldAfterPaint()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      await yieldToIdle(250)
      if (!this.isCurrent(generation) || abort.signal.aborted) return

      this.prewarmTranslator(abort.signal)
      void this.drain(generation, abort)
    } catch (error) {
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      this.clearOverlay()
      if (this.abort === abort) this.abort = null
      this.setState({ phase: 'error', error: errorMessage(error) })
    }
  }

  private async drain(generation: number, abort: AbortController): Promise<void> {
    if (this.draining) return
    this.draining = true
    const maxInFlight = this.maxPagesInFlight()
    const inFlight = new Set<Promise<void>>()
    const inFlightPages = new Set<number>()
    try {
      while (this.isCurrent(generation) && !abort.signal.aborted) {
        // Measure the DOM once per round (not per slot) to avoid a burst of
        // forced reflows on press, then fill open slots from that snapshot.
        // Pages already processing are excluded by the scheduler (pending/failed
        // only), so OCR of one page overlaps the DeepL round-trip of another.
        const layout = this.measure()
        if (!layout) {
          if (!inFlight.size) break
          await Promise.race(inFlight)
          continue
        }
        this.rebuildPlan(layout)
        const measured = this.measuredPages(layout)
        const visible = this.visibleRange(layout)
        while (inFlight.size < maxInFlight) {
          const unit = this.scheduler.next(this.units, measured, visible, this.config.resilience.maxChunkAttempts)
          if (!unit) break
          this.scheduler.markProcessing(unit.pageIndex)
          inFlightPages.add(unit.pageIndex)
          const task = this.processPage(unit, generation, abort.signal).finally(() => {
            inFlight.delete(task)
            inFlightPages.delete(unit.pageIndex)
          })
          inFlight.add(task)
        }
        if (!inFlight.size) break
        await Promise.race(inFlight)
        this.evictPages(measured, visible, inFlightPages)
      }
      await Promise.allSettled(inFlight)
      if (this.isCurrent(generation) && this.scheduler.isComplete(this.config.resilience.maxChunkAttempts) && this.chapter) {
        const { failed } = this.scheduler.progress()
        this.setState({ phase: 'done', translate: { done: this.chapter.pageCount, total: this.chapter.pageCount }, failed })
        if (this.abort === abort) this.abort = null
      }
    } finally {
      this.draining = false
    }
  }

  private maxPagesInFlight(): number {
    const caps = detectBrowserCapabilities()
    return caps.isMobile ? this.config.translator.maxPagesInFlightMobile : this.config.translator.maxPagesInFlightDesktop
  }

  private async processPage(
    unit: PageScanUnit,
    generation: number,
    signal: AbortSignal,
  ): Promise<void> {
    const chapter = this.chapter
    const pages = this.pages
    if (!chapter || !pages) return
    this.scheduler.markProcessing(unit.pageIndex)
    try {
      // Load the page and its halo neighbors so their real source sizes are
      // known. After prepare, transforms use decoded source px (DOM-independent),
      // so we only need the blobs decoded, not the DOM settled.
      for (const index of [unit.prevIndex, unit.pageIndex, unit.nextIndex]) {
        if (index !== null) await pages.read(index, signal)
      }
      throwIfAborted(signal)
      // Re-plan with real sizes so halo heights match decoded pages.
      const resolved = this.resolveUnit(unit.pageIndex) ?? unit

      const overlay = await this.pipeline.run({
        unit: resolved,
        loadPage: (index: number): Promise<LoadedPage> => pages.read(index, signal),
        sourceLanguage: chapter.sourceLanguage,
        targetLanguage: chapter.targetLanguage,
        signal,
      })
      if (!this.isCurrent(generation) || signal.aborted) return
      this.pageOverlays.set(unit.pageIndex, overlay)
      this.overlayRevision += 1
      this.scheduler.markDone(unit.pageIndex)
      this.syncOverlay()
    } catch (error) {
      if (signal.aborted || !this.isCurrent(generation)) return
      const attempts = this.scheduler.markFailed(unit.pageIndex, errorMessage(error))
      if (attempts < this.config.resilience.maxChunkAttempts) {
        await delay(this.config.resilience.backoffMs * attempts)
      }
    } finally {
      if (this.isCurrent(generation) && !signal.aborted) {
        const { done, total, failed } = this.scheduler.progress()
        this.setState({ translate: progressFromPages(done, total, chapter.pageCount), failed })
      }
    }
  }

  private rebuildPlan(layout: ChapterContentLayout): void {
    this.units = planPageScans(
      layout.pages.map(page => ({ pageIndex: page.pageIndex, source: this.pageSizeFor(page.pageIndex) ?? page.sourceSize })),
      this.config.scan,
      sourceUsesHalo(this.chapter?.sourceLanguage),
    )
    this.scheduler.reset(this.units)
  }

  private resolveUnit(pageIndex: number): PageScanUnit | null {
    return this.units.find(unit => unit.pageIndex === pageIndex) ?? null
  }

  private measure(): ChapterContentLayout | null {
    return measureLayout(this.overlays.currentHost, index => this.pageSizeFor(index))
  }

  private measuredPages(layout: ChapterContentLayout): ReadonlyMap<number, MeasuredPage> {
    return new Map(measuredPagesFromLayout(layout).map(page => [page.pageIndex, page]))
  }

  private pageSizeFor(index: number): SourcePageSize | null {
    return this.pages?.size(index) ?? null
  }

  private visibleRange(layout: ChapterContentLayout) {
    const host = this.overlays.currentHost
    if (!host) return { top: 0, bottom: 0, center: 0 }
    return visibleContentRange(host, layout.contentSize, this.config.chunk.processMarginPx)
  }

  private evictPages(measured: ReadonlyMap<number, MeasuredPage>, visible: { top: number; bottom: number }, pin?: Iterable<number>): void {
    const pages = this.pages
    if (!pages) return
    const keep = new Set<number>()
    for (const page of measured.values()) {
      if (page.domTop < visible.bottom && visible.top < page.domTop + page.domHeight) {
        keep.add(page.pageIndex)
        keep.add(page.pageIndex - 1)
        keep.add(page.pageIndex + 1)
      }
    }
    if (pin) for (const index of pin) {
      keep.add(index)
      keep.add(index - 1)
      keep.add(index + 1)
    }
    pages.evictExcept(keep)
  }

  private ensureTranslator(): Translator {
    if (this.translator && this.translatorProvider === this.provider) return this.translator
    this.disposeTranslator()
    const caps = detectBrowserCapabilities()
    const maxSessions = caps.isMobile ? this.config.translator.maxSessionsMobile : this.config.translator.maxSessionsDesktop
    this.translator = this.provider === 'google'
      ? new GoogleTranslateWeb()
      : new DeepLTranslateWeb({ maxSessions })
    this.translatorProvider = this.provider
    return this.translator
  }

  /** Open DeepL sessions ahead of the first page so it skips handshake latency. */
  private prewarmTranslator(signal: AbortSignal): void {
    const translator = this.ensureTranslator()
    if ('prewarm' in translator && typeof translator.prewarm === 'function') {
      void (translator as { prewarm(count: number, signal?: AbortSignal): Promise<void> })
        .prewarm(this.maxPagesInFlight(), signal)
        .catch(() => {})
    }
  }

  private disposeTranslator(): void {
    const translator = this.translator
    if (translator && 'close' in translator && typeof translator.close === 'function') {
      void (translator as { close(): unknown }).close()
    }
    this.translator = null
    this.translatorProvider = null
  }

  private syncOverlay(): void {
    deduplicateSeamBlocks(this.pageOverlays)
    this.overlays.update(
      this.pageOverlays,
      this.overlayRevision,
      { sourceLanguage: this.chapter?.sourceLanguage ?? null, targetLanguage: this.chapter?.targetLanguage ?? null },
    )
  }

  private clearOverlay(): void {
    this.clearOverlayKeepActive()
    this.overlays.detach()
  }

  private clearOverlayKeepActive(): void {
    this.pageOverlays = new Map()
    this.units = []
    this.scheduler.clear()
    this.pages?.clear()
    this.overlayRevision += 1
    this.overlays.detach()
  }

  private stopRun(): void {
    this.abort?.abort(new Error('translation cancelled'))
    this.abort = null
    this.draining = false
  }

  private isSameChapter(chapter: ReaderTranslationChapter): boolean {
    return this.chapter?.chapterKey === chapter.chapterKey
      && this.chapter.pageCount === chapter.pageCount
      && this.chapter.sourceLanguage === chapter.sourceLanguage
      && this.chapter.targetLanguage === chapter.targetLanguage
  }

  private isCurrent(generation: number): boolean {
    return generation === this.generation && this.active
  }

  private setState(patch: Partial<ReaderTranslationState>): void {
    this.state = { ...this.state, ...patch }
    this.emit()
  }

  private emit(): void {
    for (const listener of this.listeners) {
      try { listener(this.state) } catch {}
    }
  }
}

function progressFromPages(donePages: number, totalPages: number, pageCount: number): ReaderTranslationState['translate'] {
  if (!totalPages) return { done: 0, total: pageCount }
  return { done: Math.min(pageCount, Math.floor((donePages / totalPages) * pageCount)), total: pageCount }
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function yieldToBrowser(): Promise<void> {
  await new Promise<void>(resolve => setTimeout(resolve, 0))
}

function yieldAfterPaint(): Promise<void> {
  return new Promise<void>(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)))
}
