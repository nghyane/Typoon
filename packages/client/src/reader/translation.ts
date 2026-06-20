import { detectBrowserCapabilities, type BrowserCapabilities } from '../adapters/browserCapabilities'
import { ModelRepository } from '../adapters/ModelRepository'
import { MangaTextRegionDetector } from '../detectors/manga/MangaTextRegionDetector'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { ChapterContentLayout } from '../domain/chapterContent'
import type { ImagePixels } from '../domain/image'
import type { TextRegion } from '../domain/regions'
import type { CapabilityStatus } from '../domain/capability'
import { OrtRuntime } from '../models/OrtRuntime'
import { OrtSessionPool, type OrtProvider } from '../models/OrtSessionPool'
import type { OrtModule } from '../models/OrtBackend'
import type { ChapterOcrChunk, SourcePageSize } from '../pipeline/chapterContent'
import type { EncodedOcrImage } from '../recognizers/text'
import { attachOverlay } from '../render/overlay'
import { ensureMangaFontLoaded } from '../render/font'
import { LensTextRecognizer } from '../recognizers/lens/LensTextRecognizer'
import {
  chapterOcrChunks,
  pagesIntersectingChunk,
  rectIntersection,
} from '../pipeline/chapterContent'
import {
  emptyChapterContentOverlay,
  mergeChapterContentOverlay,
  translateChapterContentChunk,
  type ChapterContentOverlay,
} from '../pipeline/chapterContentTranslation'
import { DeepLTranslateWeb } from '../translators/deepl-web/DeepLTranslateWeb'
import ortWebgpuMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWebgpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'
import ortWasmMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'

export type ReaderPhase = 'idle' | 'loading' | 'preparing' | 'ready' | 'translating' | 'done' | 'error'

export interface ReaderTranslationState {
  phase: ReaderPhase
  prepare: { done: number; total: number; preparedPages: number }
  translate: { done: number; total: number }
  model: ReaderModelState
  sourceLanguage: string | null
  targetLanguage: string
  error?: string
}

export interface ReaderModelState {
  readonly state: 'idle' | 'resolving' | 'downloading' | 'initializing' | 'ready' | 'failed'
  readonly receivedBytes?: number
  readonly totalBytes?: number
  readonly ratio?: number
  readonly error?: string
}

type Listener = (state: ReaderTranslationState) => void

export interface ReaderTranslationChapter {
  readonly chapterKey: string
  readonly pageCount: number
  readonly readPage: (index: number, signal?: AbortSignal) => Promise<Blob>
  readonly sourceLanguage: string | null
  readonly targetLanguage: string
}

interface LoadedPage {
  readonly index: number
  readonly blob: Blob
  readonly size: SourcePageSize
}

interface CapturedOcrChunk {
  readonly encoded: EncodedOcrImage
  readonly image: ImagePixels
}

const OVERLAY_VIEWPORT_MARGIN_PX = 1400
const CHUNK_PROCESS_VIEWPORT_MARGIN_PX = 450
const DISCORD_CDN_PROXY_BASE = 'https://927251094806098001.discordsays.com/cdn/c'

const modelRepository = ModelRepository.fromHuggingFace({
  repo: 'nghyane/comic-detr',
  revision: 'v1',
  proxyBase: DISCORD_CDN_PROXY_BASE,
})
let ortRuntimePromise: Promise<ConfiguredOrtRuntime> | null = null
let textRegionDetectorPromise: Promise<TextRegionDetector> | null = null
let latestModelState: ReaderModelState = { state: 'idle' }
const modelStateListeners = new Set<(state: ReaderModelState) => void>()

interface ConfiguredOrtRuntime {
  readonly sessionPool: OrtSessionPool
  readonly providers: readonly OrtProvider[]
}

interface OrtBackend {
  readonly ort: OrtModule
  readonly wasmPaths: { readonly wasm: string; readonly mjs: string }
}

function init(pageCount: number, sourceLang: string | null, targetLang: string): ReaderTranslationState {
  return {
    phase: 'idle',
    prepare: { done: 0, total: pageCount, preparedPages: 0 },
    translate: { done: 0, total: pageCount },
    model: latestModelState,
    sourceLanguage: sourceLang,
    targetLanguage: targetLang,
  }
}

export class ReaderTranslation {
  private readonly listeners = new Set<Listener>()
  private readonly recognizer = new LensTextRecognizer()
  private translator: DeepLTranslateWeb | null = null
  private contentHost: HTMLElement | null = null
  private chapter: ReaderTranslationChapter | null = null
  private state: ReaderTranslationState = init(0, null, 'vi')
  private overlay: ChapterContentOverlay | null = null
  private pageSizes: Array<SourcePageSize | null> = []
  private readonly pageCache = new Map<number, LoadedPage>()
  private chunks: readonly ChapterOcrChunk[] = []
  private readonly processedChunks = new Set<number>()
  private readonly processingChunks = new Set<number>()
  private abort: AbortController | null = null
  private generation = 0
  private overlayRevision = 0
  private attachedOverlayKey = ''
  private active = false
  private renderFrame = 0
  private processingVisible = false
  private readonly unsubscribeModelState: () => void

  constructor() {
    this.unsubscribeModelState = subscribeModelState(model => {
      this.setState({ model })
    })
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn)
    fn(this.state)
    return () => this.listeners.delete(fn)
  }

  registerContentHost(host: HTMLElement): () => void {
    this.contentHost = host
    this.attachedOverlayKey = ''
    this.bindViewportListeners()
    this.scheduleAttachOverlay()
    return () => {
      if (this.contentHost === host) {
        this.detachOverlay(host)
        this.attachedOverlayKey = ''
        this.contentHost = null
        this.unbindViewportListeners()
      }
    }
  }

  setChapter(chapter: ReaderTranslationChapter): void {
    if (this.isSameChapter(chapter)) return
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    this.chapter = chapter
    this.active = chapter.pageCount > 0
    this.state = { ...init(chapter.pageCount, chapter.sourceLanguage, chapter.targetLanguage), phase: this.active ? 'ready' : 'idle' }
    this.emit()
    if (this.active) void ensureMangaFontLoaded()
  }

  clear(): void {
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    this.chapter = null
    this.active = false
    this.state = init(0, this.state.sourceLanguage, this.state.targetLanguage)
    this.emit()
  }

  translate(): void {
    if (!this.active || !this.chapter?.pageCount) return
    if (this.abort && !this.abort.signal.aborted) return
    const generation = ++this.generation
    void this.start(generation)
  }

  cancel(): void {
    this.generation += 1
    this.stopRun()
    this.clearOverlay()
    if (this.chapter?.pageCount) {
      this.active = true
      this.state = { ...init(this.chapter.pageCount, this.chapter.sourceLanguage, this.chapter.targetLanguage), phase: 'ready' }
    } else {
      this.active = false
      this.state = init(0, this.state.sourceLanguage, this.state.targetLanguage)
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
    this.unbindViewportListeners()
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
      error: undefined,
    })

    try {
      await yieldAfterPaint()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      this.clearOverlay()
      await yieldToBrowser()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      await ensureMangaFontLoaded()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      this.pageSizes = new Array<SourcePageSize | null>(chapter.pageCount).fill(null)
      const layout = this.measureLayout()
      if (!layout) throw new Error('chapter content host is not ready')
      this.chunks = chapterOcrChunks(layout)
      this.overlay = emptyChapterContentOverlay(layout)
      this.setState({
        phase: 'translating',
        prepare: { done: 0, total: chapter.pageCount, preparedPages: 0 },
        translate: { done: 0, total: chapter.pageCount },
      })
      await yieldAfterPaint()
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      await yieldToIdle(250)
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      this.scheduleAttachOverlay()
      this.scheduleVisibleChunkProcessing(generation)
    } catch (error) {
      if (!this.isCurrent(generation) || abort.signal.aborted) return
      this.clearOverlay()
      if (this.abort === abort) this.abort = null
      this.setState({ phase: 'error', error: errorMessage(error) })
    }
  }

  private async loadPage(chapter: ReaderTranslationChapter, index: number, signal: AbortSignal): Promise<LoadedPage> {
    const cached = this.pageCache.get(index)
    if (cached) return cached
    throwIfAborted(signal)
    const blob = await chapter.readPage(index, signal)
    const size = await readImageSize(blob)
    const page = { index, blob, size }
    this.pageCache.set(index, page)
    this.pageSizes[index] = size
    this.setState({
      prepare: { done: this.pageCache.size, total: chapter.pageCount, preparedPages: this.pageCache.size },
    })
    await yieldToBrowser()
    return page
  }

  private ensureTranslator(): DeepLTranslateWeb {
    if (this.translator) return this.translator
    const caps = detectBrowserCapabilities()
    this.translator = new DeepLTranslateWeb({ maxSessions: caps.isMobile ? 1 : 2 })
    return this.translator
  }

  private scheduleAttachOverlay = (): void => {
    if (this.renderFrame) return
    this.renderFrame = requestAnimationFrame(() => {
      this.renderFrame = 0
      this.attachVisibleOverlay()
      this.scheduleVisibleChunkProcessing(this.generation)
    })
  }

  private scheduleVisibleChunkProcessing(generation: number): void {
    if (this.processingVisible) return
    const abort = this.abort
    if (!abort || abort.signal.aborted || !this.chunks.length || !this.overlay) return
    this.processingVisible = true
    void this.processVisibleChunks(generation, abort).finally(() => {
      this.processingVisible = false
      if (this.state.phase === 'translating' && this.isCurrent(generation) && !abort.signal.aborted && this.nextChunkIndex() !== null) {
        this.scheduleVisibleChunkProcessing(generation)
      }
    })
  }

  private async processVisibleChunks(generation: number, abort: AbortController): Promise<void> {
    if (!this.isCurrent(generation) || abort.signal.aborted) return
    const next = this.nextChunkIndex()
    if (next === null) return
    this.processingChunks.add(next)
    try {
      await this.processChunk(this.chunks[next]!, generation, abort.signal)
      this.processedChunks.add(next)
      this.setState({ translate: progressFromChunks(this.processedChunks.size, this.chunks.length, this.chapter?.pageCount ?? this.chunks.length) })
    } catch (error) {
      if (!abort.signal.aborted && this.isCurrent(generation)) {
        if (this.abort === abort) this.abort = null
        this.setState({ phase: 'error', error: errorMessage(error) })
      }
      return
    } finally {
      this.processingChunks.delete(next)
    }
    if (this.processedChunks.size === this.chunks.length && this.chapter) {
      this.setState({ phase: 'done', translate: { done: this.chapter.pageCount, total: this.chapter.pageCount } })
      if (this.abort === abort) this.abort = null
    }
  }

  private nextVisibleChunkIndex(): number | null {
    const host = this.contentHost
    const layout = this.measureLayout()
    if (!host?.isConnected || !layout) return null
    this.refreshLayout(layout)
    const visible = visibleContentRange(host, layout.contentSize, CHUNK_PROCESS_VIEWPORT_MARGIN_PX)
    const candidates = this.chunks
      .map((chunk, index) => ({ chunk, index }))
      .filter(({ chunk, index }) => !this.processedChunks.has(index) && !this.processingChunks.has(index) && rangesOverlap(chunk.contentRect.y, chunk.contentRect.y + chunk.contentRect.height, visible.top, visible.bottom))
      .sort((a, b) => Math.abs(chunkCenterY(a.chunk) - visible.center) - Math.abs(chunkCenterY(b.chunk) - visible.center))
    return candidates[0]?.index ?? null
  }

  private nextChunkIndex(): number | null {
    return this.nextVisibleChunkIndex() ?? this.nextSequentialChunkIndex()
  }

  private nextSequentialChunkIndex(): number | null {
    for (let index = 0; index < this.chunks.length; index += 1) {
      if (!this.processedChunks.has(index) && !this.processingChunks.has(index)) return index
    }
    return null
  }

  private async processChunk(chunk: ChapterOcrChunk, generation: number, signal: AbortSignal): Promise<void> {
    const chapter = this.chapter
    const initialLayout = this.measureLayout()
    if (!chapter || !initialLayout || !this.overlay) return
    await this.ensurePagesForChunk(chapter, initialLayout, chunk, signal)
    await yieldAfterPaint()
    throwIfAborted(signal)
    const layout = this.measureLayout() ?? initialLayout
    this.refreshLayout(layout)
    const actualChunk = this.chunks[chunk.index] ?? chunk
    const capture = await captureChunkForOcr(index => this.loadPage(chapter, index, signal), layout, actualChunk, signal)
    await yieldAfterPaint()
    throwIfAborted(signal)
    const recognized = await this.recognizer.recognizeEncoded(capture.encoded, {
      pageIndex: actualChunk.index,
      sourceLang: chapter.sourceLanguage,
      signal,
    })
    await yieldAfterPaint()
    throwIfAborted(signal)
    const regions = await detectTextRegions(capture.image, signal)
    if (!this.isCurrent(generation)) return
    const translated = await translateChapterContentChunk({
      recognized,
      chunk: actualChunk,
      layout,
      image: capture.image,
      regions,
      translator: () => this.ensureTranslator(),
      sourceLanguage: chapter.sourceLanguage,
      targetLanguage: chapter.targetLanguage,
      signal,
    })
    if (!this.isCurrent(generation)) return
    this.overlay = mergeChapterContentOverlay(this.overlay, translated)
    this.overlayRevision += 1
    this.attachedOverlayKey = ''
    this.attachVisibleOverlay()
  }

  private attachVisibleOverlay(): void {
    const host = this.contentHost
    const overlay = this.overlay
    if (!host?.isConnected || !overlay) return

    const container = host.parentElement
    if (!container) return
    const pages = visiblePageElements(container)
    const overlayKey = `${this.overlayRevision}:${pages.map(page => page.pageIndex).join(',')}`
    if (overlayKey === this.attachedOverlayKey) return
    this.detachOverlay(host)
    for (const page of pages) {
      const items = placementItemsForPage(overlay, page.pageIndex)
      if (!items.length) continue
      const pageSize = pageSizeForOverlayPage(items, page)
      attachOverlay(page.element, {
        pageSize,
        placements: items.map(item => item.placement),
        translations: overlay.translations,
        placementMargins: items.map(item => item.margin),
        fontContextPlacements: overlay.placements,
        sourceLanguage: this.chapter?.sourceLanguage ?? null,
        targetLanguage: this.chapter?.targetLanguage ?? null,
      }, { scaleMode: 'contain' })
    }
    this.attachedOverlayKey = overlayKey
  }

  private clearOverlay(): void {
    this.overlay = null
    this.pageSizes = []
    this.pageCache.clear()
    this.chunks = []
    this.processedChunks.clear()
    this.processingChunks.clear()
    this.overlayRevision += 1
    this.attachedOverlayKey = ''
    if (this.contentHost) this.detachOverlay(this.contentHost)
  }

  private detachOverlay(host: HTMLElement): void {
    for (const child of [...host.children]) {
      if (child instanceof HTMLElement && child.dataset.typoonOverlay === 'true') child.remove()
    }
    const container = host.parentElement
    if (!container) return
    for (const page of container.querySelectorAll<HTMLElement>('[data-page-index]')) {
      for (const child of [...page.children]) {
        if (child instanceof HTMLElement && child.dataset.typoonOverlay === 'true') child.remove()
      }
    }
  }

  private measureLayout(): ChapterContentLayout | null {
    const host = this.contentHost
    const container = host?.parentElement
    if (!host?.isConnected || !container) return null

    const containerRect = container.getBoundingClientRect()
    const pages = [...container.querySelectorAll<HTMLElement>('[data-page-index]')]
      .map(element => {
        const pageIndex = Number(element.dataset.pageIndex)
        const rect = element.getBoundingClientRect()
        const width = Math.max(1, rect.width)
        const height = Math.max(1, rect.height)
        const knownSize = this.pageSizes[pageIndex] ?? this.pageCache.get(pageIndex)?.size
        return {
          pageIndex,
          sourceSize: knownSize ?? { width, height },
          contentRect: {
            x: rect.left - containerRect.left,
            y: rect.top - containerRect.top,
            width,
            height,
          },
        }
      })
      .filter(page => Number.isFinite(page.pageIndex))
      .sort((a, b) => a.pageIndex - b.pageIndex)

    if (!pages.length) return null
    const contentWidth = Math.max(1, containerRect.width)
    const contentHeight = Math.max(
      1,
      container.scrollHeight,
      ...pages.map(page => page.contentRect.y + page.contentRect.height),
    )
    return { contentSize: { width: contentWidth, height: contentHeight }, pages }
  }

  private refreshLayout(layout: ChapterContentLayout): void {
    this.chunks = chapterOcrChunks(layout)
    if (this.overlay) this.overlay = { ...this.overlay, contentSize: layout.contentSize }
  }

  private async ensurePagesForChunk(
    chapter: ReaderTranslationChapter,
    layout: ChapterContentLayout,
    chunk: ChapterOcrChunk,
    signal: AbortSignal,
  ): Promise<void> {
    for (const page of pagesIntersectingChunk(layout, chunk)) {
      await this.loadPage(chapter, page.pageIndex, signal)
    }
  }

  private bindViewportListeners(): void {
    window.addEventListener('scroll', this.scheduleAttachOverlay, true)
    window.addEventListener('resize', this.scheduleAttachOverlay)
  }

  private unbindViewportListeners(): void {
    window.removeEventListener('scroll', this.scheduleAttachOverlay, true)
    window.removeEventListener('resize', this.scheduleAttachOverlay)
    if (this.renderFrame) cancelAnimationFrame(this.renderFrame)
    this.renderFrame = 0
  }

  private stopRun(): void {
    this.abort?.abort(new Error('translation cancelled'))
    this.abort = null
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

async function captureChunkForOcr(
  loadPage: (index: number) => Promise<LoadedPage>,
  layout: ChapterContentLayout,
  chunk: ChapterOcrChunk,
  signal: AbortSignal,
): Promise<CapturedOcrChunk> {
  const width = Math.max(1, Math.ceil(chunk.contentRect.width))
  const height = Math.max(1, Math.ceil(chunk.contentRect.height))
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d canvas unavailable')
  ctx.fillStyle = '#fff'
  ctx.fillRect(0, 0, width, height)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'

  for (const page of pagesIntersectingChunk(layout, chunk)) {
    throwIfAborted(signal)
    const input = await loadPage(page.pageIndex)
    const intersection = rectIntersection(page.contentRect, chunk.contentRect)
    if (!intersection) continue

    const bitmap = await createImageBitmap(input.blob)
    try {
      const scale = page.contentRect.width / Math.max(1, input.size.width)
      const sx = (intersection.x - page.contentRect.x) / scale
      const sy = (intersection.y - page.contentRect.y) / scale
      const sw = intersection.width / scale
      const sh = intersection.height / scale
      const dx = intersection.x - chunk.contentRect.x
      const dy = intersection.y - chunk.contentRect.y
      ctx.drawImage(bitmap, sx, sy, sw, sh, dx, dy, intersection.width, intersection.height)
    } finally {
      bitmap.close()
    }
    await yieldToBrowser()
  }

  const pixels = ctx.getImageData(0, 0, width, height)
  const blob = await canvasToOcrBlob(canvas)
  return {
    encoded: {
      bytes: new Uint8Array(await blob.arrayBuffer()),
      width,
      height,
      originalWidth: width,
      originalHeight: height,
    },
    image: { width, height, data: pixels.data },
  }
}

interface PageOverlayHost {
  readonly pageIndex: number
  readonly element: HTMLElement
}

function visiblePageElements(container: HTMLElement): PageOverlayHost[] {
  const pages = [...container.querySelectorAll<HTMLElement>('[data-page-index]')]
  return pages
    .map(element => ({ element, pageIndex: Number(element.dataset.pageIndex) }))
    .filter(page => Number.isFinite(page.pageIndex) && pageVisible(page.element, OVERLAY_VIEWPORT_MARGIN_PX))
}

function pageVisible(element: HTMLElement, marginPx: number): boolean {
  const rect = element.getBoundingClientRect()
  return rect.bottom >= -marginPx && rect.top <= window.innerHeight + marginPx
}

function placementItemsForPage(overlay: ChapterContentOverlay, pageIndex: number): Array<{ placement: ChapterContentOverlay['placements'][number]; margin: ChapterContentOverlay['placementMargins'][number] }> {
  return overlay.placements
    .map((placement, index) => ({ placement, margin: overlay.placementMargins[index]! }))
    .filter(item => item.placement.pageIndex === pageIndex)
}

function pageSizeForOverlayPage(
  items: readonly { readonly placement: ChapterContentOverlay['placements'][number] }[],
  page: PageOverlayHost,
): readonly [number, number] {
  const fromPlacement = items.find(item => item.placement.pageSize[0] > 0 && item.placement.pageSize[1] > 0)?.placement.pageSize
  if (fromPlacement) return fromPlacement
  const rect = page.element.getBoundingClientRect()
  return [Math.max(1, rect.width), Math.max(1, rect.height)]
}

function visibleContentRange(
  host: HTMLElement,
  contentSize: ChapterContentOverlay['contentSize'],
  marginPx: number,
): { top: number; bottom: number; center: number } {
  const rect = host.getBoundingClientRect()
  const scale = rect.width / Math.max(1, contentSize.width)
  if (!Number.isFinite(scale) || scale <= 0) return { top: 0, bottom: 0, center: 0 }
  const top = Math.max(0, (-rect.top - marginPx) / scale)
  const bottom = Math.min(contentSize.height, (window.innerHeight - rect.top + marginPx) / scale)
  return { top, bottom, center: (top + bottom) / 2 }
}

function rangesOverlap(aTop: number, aBottom: number, bTop: number, bBottom: number): boolean {
  return aTop < bBottom && bTop < aBottom
}

function chunkCenterY(chunk: ChapterOcrChunk): number {
  return chunk.contentRect.y + chunk.contentRect.height / 2
}

async function readImageSize(blob: Blob): Promise<SourcePageSize> {
  const bitmap = await createImageBitmap(blob)
  try {
    return { width: bitmap.width, height: bitmap.height }
  } finally {
    bitmap.close()
  }
}

function canvasToOcrBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('failed to encode chapter chunk')), 'image/jpeg', 0.92)
  })
}

function progressFromChunks(doneChunks: number, totalChunks: number, pageCount: number): ReaderTranslationState['translate'] {
  if (!totalChunks) return { done: 0, total: pageCount }
  return { done: Math.min(pageCount, Math.floor((doneChunks / totalChunks) * pageCount)), total: pageCount }
}

async function detectTextRegions(image: ImagePixels, signal: AbortSignal): Promise<readonly TextRegion[]> {
  const detector = await defaultTextRegionDetector(signal)
  return detector.detectTextRegions(image, { signal })
}

function defaultTextRegionDetector(signal: AbortSignal): Promise<TextRegionDetector> {
  if (!textRegionDetectorPromise) {
    textRegionDetectorPromise = createTextRegionDetector(signal).catch(error => {
      textRegionDetectorPromise = null
      throw error
    })
  }
  return textRegionDetectorPromise
}

async function createTextRegionDetector(signal: AbortSignal): Promise<TextRegionDetector> {
  throwIfAborted(signal)
  const caps = detectBrowserCapabilities()
  const ortRuntime = await configureOrtRuntime(caps)
  const model = await modelRepository.model(caps.modelHint.modelId)
  const detector = new MangaTextRegionDetector({
    model,
    sessionPool: ortRuntime.sessionPool,
    preferredProviders: ortRuntime.providers,
  })
  detector.subscribeStatus(status => publishModelState(capabilityToModelState(status)))
  publishModelState(capabilityToModelState(detector.status()))
  await detector.ensureReady({ signal })
  return detector
}

function subscribeModelState(listener: (state: ReaderModelState) => void): () => void {
  modelStateListeners.add(listener)
  listener(latestModelState)
  return () => modelStateListeners.delete(listener)
}

function publishModelState(state: ReaderModelState): void {
  latestModelState = state
  for (const listener of modelStateListeners) {
    try { listener(state) } catch {}
  }
}

function capabilityToModelState(status: CapabilityStatus): ReaderModelState {
  if (status.state === 'downloading') {
    return {
      state: 'downloading',
      receivedBytes: status.progress.receivedBytes,
      totalBytes: status.progress.totalBytes,
      ratio: status.progress.ratio,
    }
  }
  if (status.state === 'failed') return { state: 'failed', error: errorMessage(status.error) }
  return { state: status.state }
}

function configureOrtRuntime(caps: BrowserCapabilities): Promise<ConfiguredOrtRuntime> {
  if (!ortRuntimePromise) {
    ortRuntimePromise = createOrtRuntime(caps).catch(error => {
      ortRuntimePromise = null
      throw error
    })
  }
  return ortRuntimePromise
}

async function createOrtRuntime(caps: BrowserCapabilities): Promise<ConfiguredOrtRuntime> {
  const backend = await loadOrtBackend(caps.modelHint.preferredProvider)
  const runtime = new OrtRuntime(backend.ort)
  runtime.configure({
    logLevel: 'fatal',
    wasmPaths: backend.wasmPaths,
    wasmNumThreads: caps.modelHint.wasmNumThreads,
  })
  return {
    sessionPool: new OrtSessionPool(backend.ort),
    providers: preferredProviders(caps.modelHint.preferredProvider),
  }
}

async function loadOrtBackend(preferredProvider: OrtProvider): Promise<OrtBackend> {
  if (preferredProvider === 'webgpu') {
    return {
      ort: (await import('onnxruntime-web/webgpu')) as OrtModule,
      wasmPaths: absoluteWasmPaths(ortWebgpuWasmUrl, ortWebgpuMjsUrl),
    }
  }
  return {
    ort: await import('onnxruntime-web/wasm'),
    wasmPaths: absoluteWasmPaths(ortWasmUrl, ortWasmMjsUrl),
  }
}

function absoluteWasmPaths(wasm: string, mjs: string): OrtBackend['wasmPaths'] {
  return {
    wasm: new URL(wasm, window.location.href).href,
    mjs: new URL(mjs, window.location.href).href,
  }
}

function preferredProviders(preferred: OrtProvider): readonly OrtProvider[] {
  return preferred === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
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
