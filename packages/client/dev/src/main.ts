import {
  DeepLTranslateWeb,
  GoogleTranslateWeb,
  LensTextRecognizer,
  MangaTextRegionDetector,
  MainThreadVisionRuntime,
  ModelRepository,
  TranslationRuntime,
  attachOverlay,
  detectBrowserCapabilities,
  type CapabilityStatus,
  type PageDocumentSource,
  type PageOverlay,
  type PipelineConcurrency,
  type Translator,
  ensureMangaFontLoaded,
  createDebugLayer,
  type OverlayDebugOptions,
} from '../../src/index'
import { OrtRuntime } from '../../src/models/OrtRuntime'
import { OrtSessionPool, type OrtProvider } from '../../src/models/OrtSessionPool'
import type { OrtModule } from '../../src/models/OrtBackend'
import ortWebgpuMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWebgpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'
import ortWasmMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import './style.css'

const chapterPages = Array.from({ length: 20 }, (_, index) => `/chapter-20/${String(index + 1).padStart(3, '0')}.jpg`)

const app = document.querySelector<HTMLDivElement>('#app')!

const browserCaps = detectBrowserCapabilities()

app.innerHTML = `
  <section class="shell">
    <aside class="panel">
      <p class="eyebrow">@typoon/client</p>
      <h1>Local overlay dev reader</h1>
      <p class="muted">
        OCR runs immediately via Lens while Comic-DETR model downloads in background.<br />
        Model auto-select: ${browserCaps.supportsStableWebGpu ? 'WebGPU (161 MB FP32)' : 'wasm (11 MB INT8)'}
        ${browserCaps.isSafari ? '— Safari: single-thread wasm' : ''}
      </p>
      <section class="panel-section" aria-labelledby="translationSettingsTitle">
        <div class="section-heading">
          <h2 id="translationSettingsTitle">Translation</h2>
          <span>en → vi</span>
        </div>
        <label class="field">
          <span>Translator</span>
          <select id="translator">
            <option value="deepl" selected>DeepL Web</option>
            <option value="google">Google Translate</option>
          </select>
        </label>
        <label id="deeplEndpointField" class="field" hidden>
          <span>DeepL endpoint</span>
          <input id="deeplEndpoint" type="text" value="" placeholder="default CDN proxy" spellcheck="false" />
          <small>Blank uses the production CDN proxy for ita-free.www.deepl.com/v2.</small>
        </label>
      </section>
      <button id="run" class="primary-action">Translate 20-page chapter</button>
      <section class="panel-section" aria-labelledby="modelStatusTitle">
        <div class="section-heading">
          <h2 id="modelStatusTitle">Models</h2>
        </div>
        <button id="prepareModels" type="button">Download / prepare models</button>
        <pre id="modelStatus">model idle</pre>
      </section>
      <section class="panel-section" aria-labelledby="overlayDebugTitle">
        <div class="section-heading">
          <h2 id="overlayDebugTitle">Overlay debug</h2>
        </div>
        <div class="check-list" aria-label="overlay debug controls">
          <label><input id="useOnnx" type="checkbox" checked /> ONNX detect regions</label>
          <label><input id="erase" type="checkbox" checked /> erase flat-fill</label>
          <label><input id="debugDrawable" type="checkbox" /> drawable regions</label>
          <label><input id="debugTextBoxes" type="checkbox" /> OCR boxes</label>
          <label><input id="debugBounds" type="checkbox" /> layout bounds</label>
          <label><input id="debugLabels" type="checkbox" /> labels</label>
        </div>
      </section>
      <section class="panel-section" aria-labelledby="runOutputTitle">
        <div class="section-heading">
          <h2 id="runOutputTitle">Run output</h2>
        </div>
        <pre id="status">status idle</pre>
        <pre id="log">idle</pre>
      </section>
    </aside>
    <section id="reader" class="reader">
      ${chapterPages.map((src, index) => `
        <div class="page-host" data-page-index="${index}">
          <img class="page-image" src="${src}" alt="MangaDex chapter page ${index + 1}" loading="eager" />
        </div>
      `).join('')}
    </section>
  </section>
`

const run = document.querySelector<HTMLButtonElement>('#run')!
const status = document.querySelector<HTMLPreElement>('#status')!
const log = document.querySelector<HTMLPreElement>('#log')!
const prepareModels = document.querySelector<HTMLButtonElement>('#prepareModels')!
const modelStatus = document.querySelector<HTMLPreElement>('#modelStatus')!
const reader = document.querySelector<HTMLDivElement>('#reader')!
const erase = document.querySelector<HTMLInputElement>('#erase')!
const useOnnx = document.querySelector<HTMLInputElement>('#useOnnx')!
const debugDrawable = document.querySelector<HTMLInputElement>('#debugDrawable')!
const debugTextBoxes = document.querySelector<HTMLInputElement>('#debugTextBoxes')!
const debugBounds = document.querySelector<HTMLInputElement>('#debugBounds')!
const debugLabels = document.querySelector<HTMLInputElement>('#debugLabels')!
const translatorSelect = document.querySelector<HTMLSelectElement>('#translator')!
const deeplEndpoint = document.querySelector<HTMLInputElement>('#deeplEndpoint')!
const deeplEndpointField = document.querySelector<HTMLElement>('#deeplEndpointField')!

const renderedPages = new Map<number, PageOverlay>()

const pageSource: PageDocumentSource = {
  pageCount: chapterPages.length,
  async readPage(index, signal) {
    const src = chapterPages[index]
    if (!src) throw new RangeError(`page ${index} out of range`)
    const res = await fetch(src, { signal })
    if (!res.ok) throw new Error(`page ${index} fetch failed: ${res.status}`)
    const blob = await res.blob()
    const image = imageElement(index)
    const size = image.naturalWidth && image.naturalHeight
      ? { width: image.naturalWidth, height: image.naturalHeight }
      : undefined
    return { index, blob, size }
  },
}

const models = ModelRepository.fromHuggingFace({
  repo: 'nghyane/comic-detr',
  revision: 'v1',
})

const recognizer = new LensTextRecognizer()

const ortBackend = await loadOrtBackend(browserCaps.modelHint.preferredProvider)
const ortRuntime = new OrtRuntime(ortBackend.ort)
ortRuntime.configure({
  logLevel: 'fatal',
  wasmPaths: ortBackend.wasmPaths,
  wasmNumThreads: browserCaps.modelHint.wasmNumThreads,
})
const sessionPool = new OrtSessionPool(ortBackend.ort)

const comicDetrModel = await models.model(browserCaps.modelHint.modelId)
comicDetrModel.subscribeStatus(renderModelStatus)

const detector = new MangaTextRegionDetector({
  model: comicDetrModel,
  sessionPool,
  preferredProviders: preferredProviders(browserCaps.modelHint.preferredProvider),
})
detector.subscribeStatus(renderModelStatus)

const pipelineConcurrency: PipelineConcurrency = {
  load: browserCaps.isMobile ? 2 : 4,
  prepare: 1,
  ocr: browserCaps.isMobile ? 2 : 3,
  detect: 1,
  translate: browserCaps.isMobile ? 3 : 4,
  compose: browserCaps.isMobile ? 2 : 3,
}

status.textContent = 'status idle'
syncTranslatorControls()

useOnnx.addEventListener('change', () => {
  prepareModels.disabled = !useOnnx.checked
  modelStatus.textContent = useOnnx.checked ? 'model idle' : 'ONNX disabled'
})

prepareModels.addEventListener('click', async () => {
  prepareModels.disabled = true
  try {
    await detector.ensureReady()
  } catch (error) {
    modelStatus.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  } finally {
    prepareModels.disabled = false
  }
})

run.addEventListener('click', async () => {
  run.disabled = true
  prepareModels.disabled = true
  const translator = createTranslator()
  const runtime = createRuntime(translator)
  clearOverlays()
  clearPendingOverlays()
  clearFailedOverlays()
  renderedPages.clear()
  log.textContent = `running chapter with ${translator.name}…`
  const startedAt = performance.now()
  try {
    await ensureMangaFontLoaded()
    markPendingPages()
    const translationRun = runtime.createTranslationRun(pageSource, {
      sourceLanguage: 'en',
      targetLanguage: 'vi',
      scope: 'all',
      priority: { aroundPageIndex: 0 },
      preparation: { type: 'identity' },
    })
    translationRun.subscribe(event => {
      if (event.type === 'page-overlay') {
        renderedPages.set(event.overlay.pageIndex, event.overlay)
        setPagePending(event.overlay.pageIndex, false)
        attachPageOverlay(event.overlay)
        logProgress(renderedPages.size, startedAt)
        return
      }
      if (event.type === 'page-status' && event.status === 'error') {
        setPagePending(event.pageIndex, false)
        setPageFailed(event.pageIndex, event.error)
        return
      }
      if (event.type === 'progress') {
        status.textContent = `status running ${event.progress.done}/${event.progress.total}`
      }
    })
    translationRun.start()
    await translationRun.done
    status.textContent = `status done ${renderedPages.size}/${pageSource.pageCount}`
    logProgress(renderedPages.size, startedAt)
  } catch (error) {
    log.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  } finally {
    runtime.dispose()
    await closeTranslator(translator)
    run.disabled = false
    prepareModels.disabled = false
  }
})

translatorSelect.addEventListener('change', syncTranslatorControls)

erase.addEventListener('change', renderRenderedPages)

for (const input of [debugDrawable, debugTextBoxes, debugBounds, debugLabels]) {
  input.addEventListener('change', renderDebugLayers)
}

function renderRenderedPages(): void {
  clearOverlays()
  for (const [, page] of renderedPages) attachPageOverlay(page)
  logProgress(renderedPages.size, null)
}

function renderDebugLayers(): void {
  const debug = overlayDebugOptions()
  const showDebug = hasDebugOptions(debug)

  for (const [index, page] of renderedPages) {
    const host = pageHost(index)
    const overlay = host.querySelector<HTMLElement>('[data-typoon-overlay="true"]')
    if (!overlay) {
      attachPageOverlay(page)
      continue
    }

    overlay.querySelectorAll('[data-typoon-debug-layer="true"]').forEach(layer => layer.remove())
    if (showDebug) overlay.appendChild(createDebugLayer(page.placements, overlayPageSize(page), debug))
  }
}

function overlayOptions() {
  return {
    eraseStrategy: erase.checked ? 'flat-fill' : 'none',
    debug: overlayDebugOptions(),
  } as const
}

function overlayDebugOptions(): OverlayDebugOptions {
  return {
    showDrawable: debugDrawable.checked,
    showTextBoxes: debugTextBoxes.checked,
    showTextBounds: debugBounds.checked,
    showLabels: debugLabels.checked,
  }
}

function hasDebugOptions(options: OverlayDebugOptions): boolean {
  return !!(options.showDrawable || options.showTextBoxes || options.showTextBounds || options.showLabels)
}

function createRuntime(translator: Translator): TranslationRuntime {
  return new TranslationRuntime({
    vision: new MainThreadVisionRuntime({
      detector: useOnnx.checked ? detector : undefined,
    }),
    recognizer,
    translator,
    concurrency: pipelineConcurrency,
  })
}

function createTranslator(): Translator {
  if (translatorSelect.value === 'deepl') {
    const endpoint = deeplEndpoint.value.trim()
    return new DeepLTranslateWeb({
      ...(endpoint ? { endpoint } : {}),
      maxSessions: pipelineConcurrency.translate,
    })
  }
  return new GoogleTranslateWeb({ maxConcurrency: pipelineConcurrency.translate })
}

function renderModelStatus(status: CapabilityStatus): void {
  modelStatus.textContent = JSON.stringify({
    name: status.name,
    state: status.state,
    progress: status.state === 'downloading' ? {
      received: formatBytes(status.progress.receivedBytes),
      total: status.progress.totalBytes ? formatBytes(status.progress.totalBytes) : undefined,
      percent: status.progress.ratio === undefined ? undefined : `${Math.round(status.progress.ratio * 100)}%`,
    } : undefined,
    error: status.state === 'failed' ? errorMessage(status.error) : undefined,
  }, null, 2)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KiB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`
}

async function closeTranslator(translator: Translator): Promise<void> {
  if ('close' in translator && typeof translator.close === 'function') {
    try { await translator.close() } catch { /* best-effort */ }
  }
}

function syncTranslatorControls(): void {
  const isDeepL = translatorSelect.value === 'deepl'
  deeplEndpointField.hidden = !isDeepL
  deeplEndpoint.disabled = !isDeepL
}

function clearOverlays(): void {
  reader.querySelectorAll('[data-typoon-overlay="true"]').forEach(overlay => overlay.remove())
}

function markPendingPages(): void {
  for (let index = 0; index < pageSource.pageCount; index++) {
    if (!renderedPages.has(index)) setPagePending(index, true)
  }
}

function setPagePending(index: number, pending: boolean): void {
  const host = pageHost(index)
  host.querySelector('[data-dev-pending-overlay="true"]')?.remove()
  if (!pending) return
  const overlay = document.createElement('div')
  overlay.dataset.devPendingOverlay = 'true'
  overlay.innerHTML = '<span class="pending-spinner" aria-hidden="true"></span><span>Translating…</span>'
  host.appendChild(overlay)
}

function clearPendingOverlays(): void {
  reader.querySelectorAll('[data-dev-pending-overlay="true"]').forEach(overlay => overlay.remove())
}

function clearFailedOverlays(): void {
  reader.querySelectorAll('[data-dev-failed-overlay="true"]').forEach(overlay => overlay.remove())
}

function setPageFailed(index: number, error: Error | undefined): void {
  const host = pageHost(index)
  host.querySelector('[data-dev-failed-overlay="true"]')?.remove()
  const overlay = document.createElement('div')
  overlay.dataset.devFailedOverlay = 'true'
  overlay.textContent = error?.message ?? 'Translation failed'
  host.appendChild(overlay)
}

function attachPageOverlay(page: PageOverlay): void {
  const host = pageHost(page.pageIndex)
  host.querySelector('[data-typoon-overlay="true"]')?.remove()
  attachOverlay(host, {
    pageSize: [page.pageSize.width, page.pageSize.height],
    placements: page.placements,
    translations: page.translations,
    placementMargins: page.placementMargins,
  }, overlayOptions())
}

function overlayPageSize(page: PageOverlay): readonly [number, number] {
  return [page.pageSize.width, page.pageSize.height]
}

interface DevOrtBackend {
  readonly ort: OrtModule
  readonly wasmPaths: { readonly wasm: string; readonly mjs: string }
}

async function loadOrtBackend(preferredProvider: OrtProvider): Promise<DevOrtBackend> {
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

function absoluteWasmPaths(wasm: string, mjs: string): DevOrtBackend['wasmPaths'] {
  return {
    wasm: new URL(wasm, window.location.href).href,
    mjs: new URL(mjs, window.location.href).href,
  }
}

function preferredProviders(preferred: OrtProvider): readonly OrtProvider[] {
  return preferred === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']
}

function imageElement(index: number): HTMLImageElement {
  const image = pageHost(index).querySelector<HTMLImageElement>('.page-image')
  if (!image) throw new RangeError(`page image not found: ${index}`)
  return image
}

function pageHost(index: number): HTMLElement {
  const host = reader.querySelector<HTMLElement>(`[data-page-index="${index}"]`)
  if (!host) throw new RangeError(`page host not found: ${index}`)
  return host
}

function logProgress(done: number, startedAt: number | null): void {
  const elapsedMs = startedAt === null ? null : Math.round(performance.now() - startedAt)
  log.textContent = JSON.stringify({
    progress: {
      done,
      total: pageSource.pageCount,
      elapsedMs,
    },
    metrics: collectTextMetrics(),
    pages: [...renderedPages].map(([pageIndex, page]) => ({
      pageIndex,
      placementCount: page.placements.length,
      translationCount: page.translations.length,
      placements: page.placements.map(placement => ({
        id: placement.id,
        role: placement.role,
        bbox: placement.bbox,
        rotationDeg: placement.rotationDeg,
        textBoxes: placement.textBoxes,
        fontHint: placement.fontHint,
      })),
      translations: page.translations,
    })),
  }, null, 2)
}

function collectTextMetrics(): unknown[] {
  return [...reader.querySelectorAll<HTMLElement>('[data-typoon-text="true"]')].map(el => ({
    page: Number(el.closest<HTMLElement>('[data-page-index]')?.dataset.pageIndex ?? -1),
    id: el.dataset.placementId,
    role: el.dataset.role,
    font: Number(el.dataset.fontSizePx),
    sourceFont: Number(el.dataset.sourceFontPx),
    roleMedian: Number(el.dataset.roleMedianFontPx),
    targetFont: Number(el.dataset.targetFontPx),
    fontIntent: el.dataset.fontIntentReason,
    fit: el.dataset.fitReason,
    layout: el.dataset.layoutCandidate,
    expansion: el.dataset.expansionReason ?? 'none',
    margins: el.dataset.margins ?? '',
    baseRect: el.dataset.baseRect ?? '',
    safeBounds: el.dataset.safeBounds ?? '',
    direction: el.dataset.direction,
    directionReason: el.dataset.directionReason,
    maxDom: Number(el.dataset.maxDomFitPx),
    cap: el.dataset.capReason,
    overflow: el.dataset.overflow === 'true',
    rot: Number(el.dataset.rotationDeg),
    box: {
      w: Math.round(el.getBoundingClientRect().width),
      h: Math.round(el.getBoundingClientRect().height),
    },
  }))
}
