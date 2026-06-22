// reader/translation.ts — ReaderTranslation controller (public API).
//
// Thin orchestrator composed of focused modules:
//   - TranslationStore-like state held here, emitted with stable shape
//   - ChunkScheduler   : stable chunk identity + viewport-aware ordering
//   - ChunkPipeline    : pure capture → OCR → detect → translate
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
import { PagePipeline } from './pagePipeline'
import { PageScheduler } from './pageScheduler'
import { planPageScans, measuredPagesFromLayout, type MeasuredPage } from './pageScan'
import { measureLayout, visibleContentRange } from './chunkCapture'
import { OverlayManager } from './overlayManager'
import { PageProvider, type LoadedPage } from './pageProvider'
import { defaultTranslationConfig, type TranslationConfig } from './translationConfig'
import {
  prewarmTextRegionDetector,
  subscribeModelState,
  type ReaderModelState,
} from './visionRuntime'

export type ReaderPhase = 'idle' | 'loading' | 'ready' | 'translating' | 'done' | 'error'

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
}

export interface ReaderTranslationOptions {
  readonly config?: TranslationConfig
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
  private readonly overlays: OverlayManager
  private readonly scheduler = new PageScheduler()
  private readonly config: TranslationConfig
  private readonly pipeline: PagePipeline

  private translator: DeepLTranslateWeb | null = null
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
    this.overlays = new OverlayManager(this.config.chunk.overlayMarginPx)
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
    void this.translator?.close()
    this.translator = null
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
    try {
      while (this.isCurrent(generation) && !abort.signal.aborted) {
        const layout = this.measure()
        if (!layout) break
        this.rebuildPlan(layout)
        const measured = this.measuredPages(layout)
        const visible = this.visibleRange(layout)
        const unit = this.scheduler.next(this.units, measured, visible, this.config.resilience.maxChunkAttempts)
        if (!unit) break
        await this.processPage(unit, generation, abort.signal)
        this.evictPages(measured, visible)
      }
      if (this.isCurrent(generation) && this.scheduler.isComplete(this.config.resilience.maxChunkAttempts) && this.chapter) {
        const { failed } = this.scheduler.progress()
        this.setState({ phase: 'done', translate: { done: this.chapter.pageCount, total: this.chapter.pageCount }, failed })
        if (this.abort === abort) this.abort = null
      }
    } finally {
      this.draining = false
    }
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

  private evictPages(measured: ReadonlyMap<number, MeasuredPage>, visible: { top: number; bottom: number }): void {
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
    pages.evictExcept(keep)
  }

  private ensureTranslator(): DeepLTranslateWeb {
    if (this.translator) return this.translator
    const caps = detectBrowserCapabilities()
    this.translator = new DeepLTranslateWeb({
      maxSessions: caps.isMobile ? this.config.translator.maxSessionsMobile : this.config.translator.maxSessionsDesktop,
    })
    return this.translator
  }

  private syncOverlay(): void {
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

function yieldToIdle(timeoutMs: number): Promise<void> {
  const win = window as Window & {
    requestIdleCallback?: (callback: () => void, options?: { timeout?: number }) => number
  }
  if (typeof win.requestIdleCallback === 'function') {
    return new Promise<void>(resolve => { win.requestIdleCallback?.(() => resolve(), { timeout: timeoutMs }) })
  }
  return yieldToBrowser()
}

function yieldAfterPaint(): Promise<void> {
  return new Promise<void>(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)))
}

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
