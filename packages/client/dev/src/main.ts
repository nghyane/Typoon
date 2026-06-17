import {
  DeepLTranslateWeb,
  DomOverlayTarget,
  DomPageSource,
  GoogleTranslateWeb,
  LensTextRecognizer,
  MangaTextRegionDetector,
  ModelRepository,
  TranslationEngine,
  type CapabilityStatus,
  type RenderedPage,
  type Translator,
  ensureMangaFontLoaded,
} from '../../src/index'
import { OrtRuntime } from '../../src/models/OrtRuntime'
import { OrtSessionPool } from '../../src/models/OrtSessionPool'
import ortWebgpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'
import './style.css'

const comicDetrModelUrl = new URL('../assets/comic-detr-v4s-webgpu.onnx', import.meta.url).href
const ortWasmUrl = new URL(ortWebgpuWasmUrl, window.location.href).href
const chapterPages = Array.from({ length: 20 }, (_, index) => `/chapter-20/${String(index + 1).padStart(3, '0')}.jpg`)

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <section class="shell">
    <aside class="panel">
      <p class="eyebrow">@typoon/client</p>
      <h1>Local overlay dev reader</h1>
      <p class="muted">
        Client-only pipeline: browser encodes the page, recognizes text with Lens, detects regions locally, then translates with the selected web translator.
      </p>
      <section class="panel-section" aria-labelledby="translationSettingsTitle">
        <div class="section-heading">
          <h2 id="translationSettingsTitle">Translation</h2>
          <span>en → vi</span>
        </div>
        <label class="field">
          <span>Translator</span>
          <select id="translator">
            <option value="google">Google Translate</option>
            <option value="deepl">DeepL Web</option>
          </select>
        </label>
        <label id="deeplEndpointField" class="field" hidden>
          <span>DeepL endpoint</span>
          <input id="deeplEndpoint" type="text" value="/deepl/v2" spellcheck="false" />
          <small>Local dev proxies /deepl to the configured DeepL web proxy.</small>
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
const pages = new DomPageSource(reader, { imageSelector: '.page-image' })
const overlays = new DomOverlayTarget(reader, { imageSelector: '.page-image', hostSelector: '.page-host' })
const erase = document.querySelector<HTMLInputElement>('#erase')!
const debugDrawable = document.querySelector<HTMLInputElement>('#debugDrawable')!
const debugTextBoxes = document.querySelector<HTMLInputElement>('#debugTextBoxes')!
const debugBounds = document.querySelector<HTMLInputElement>('#debugBounds')!
const debugLabels = document.querySelector<HTMLInputElement>('#debugLabels')!
const translatorSelect = document.querySelector<HTMLSelectElement>('#translator')!
const deeplEndpoint = document.querySelector<HTMLInputElement>('#deeplEndpoint')!
const deeplEndpointField = document.querySelector<HTMLElement>('#deeplEndpointField')!

const renderedPages = new Map<number, RenderedPage>()

const models = new ModelRepository({
  manifest: {
    version: 'dev-local',
    models: {
      comicDetr: {
        id: 'comic-detr-v4s-webgpu',
        version: 'dev-local',
        url: comicDetrModelUrl,
        sha256: '',
        sizeBytes: 168874314,
        inputSize: 640,
        executionProviders: ['webgpu', 'wasm'],
      },
    },
  },
})

const recognizer = new LensTextRecognizer()
const comicDetrModel = models.model('comicDetr')

const ortRuntime = new OrtRuntime()
ortRuntime.configure({
  logLevel: 'fatal',
  wasmPaths: { wasm: ortWasmUrl },
  wasmNumThreads: crossOriginIsolated
    ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 1))
    : 1,
})
const sessionPool = new OrtSessionPool()

const detector = new MangaTextRegionDetector({
  model: comicDetrModel,
  sessionPool,
})
const scheduler = {
  concurrency: { pages: 4, recognize: 4, detect: 1, translate: 4 },
  retry: { recognize: { attempts: 2 }, translate: { attempts: 2 } },
}

status.textContent = 'status idle'
comicDetrModel.subscribeStatus(renderModelStatus)
detector.subscribeStatus(renderModelStatus)
syncTranslatorControls()

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
  const engine = createEngine(translator)
  clearOverlays()
  clearPendingOverlays()
  clearFailedOverlays()
  renderedPages.clear()
  log.textContent = `running chapter with ${translator.name}…`
  try {
    await ensureMangaFontLoaded()
    await detector.ensureReady()
    markPendingPages()
    const segmentRun = engine.translateSegment({
      source: pages,
      sourceLang: 'en',
      targetLang: 'vi',
    })
    segmentRun.onDisplay(page => {
      renderedPages.set(page.pageIndex, page)
      setPagePending(page.pageIndex, false)
      overlays.attach(page.pageIndex, page, overlayOptions())
    })
    segmentRun.start()
    await segmentRun.done
  } catch (error) {
    log.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  } finally {
    await closeTranslator(translator)
    run.disabled = false
    prepareModels.disabled = false
  }
})

translatorSelect.addEventListener('change', syncTranslatorControls)

for (const input of [erase, debugDrawable, debugTextBoxes, debugBounds, debugLabels]) {
  input.addEventListener('change', renderRenderedPages)
}

function renderRenderedPages(): void {
  clearOverlays()
  for (const [index, page] of renderedPages) overlays.attach(index, page, overlayOptions())
  logProgress(renderedPages.size, null)
}

function overlayOptions() {
  return {
    eraseStrategy: erase.checked ? 'flat-fill' : 'none',
    debug: {
      showDrawable: debugDrawable.checked,
      showTextBoxes: debugTextBoxes.checked,
      showTextBounds: debugBounds.checked,
      showLabels: debugLabels.checked,
    },
  } as const
}

function createEngine(translator: Translator): TranslationEngine {
  return new TranslationEngine({
    sourceLang: 'en',
    targetLang: 'vi',
    recognizer,
    detector,
    translator,
    scheduler,
  })
}

function createTranslator(): Translator {
  if (translatorSelect.value === 'deepl') {
    return new DeepLTranslateWeb({ endpoint: deeplEndpoint.value.trim() || '/deepl/v2' })
  }
  return new GoogleTranslateWeb()
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
  if ('close' in translator && typeof translator.close === 'function') await translator.close()
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
  for (let index = 0; index < pages.pageCount; index++) {
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
      total: pages.pageCount,
      elapsedMs,
    },
    metrics: collectTextMetrics(),
    pages: [...renderedPages].map(([pageIndex, page]) => ({
      pageIndex,
      phase: page.phase,
      placementCount: page.placements.length,
      translationCount: page.translations.length,
      placements: page.placements.map(placement => ({
        id: placement.id,
        role: placement.role,
        text: sourceTextForPlacement(page, placement.sourceUnitIds),
        bbox: placement.bbox,
        rotationDeg: placement.rotationDeg,
        textBoxes: placement.textBoxes,
        fontHint: placement.fontHint,
      })),
      translations: page.translations,
    })),
  }, null, 2)
}

function sourceTextForPlacement(page: RenderedPage, sourceUnitIds: readonly string[]): string {
  const byId = new Map(page.textUnits.map(unit => [unit.id, unit.sourceText]))
  return sourceUnitIds.map(id => byId.get(id)).filter(Boolean).join('\n')
}

function collectTextMetrics(): unknown[] {
  return [...reader.querySelectorAll<HTMLElement>('[data-typoon-text="true"]')].map(el => ({
    page: Number(el.closest<HTMLElement>('[data-page-index]')?.dataset.pageIndex ?? -1),
    id: el.dataset.placementId,
    role: el.dataset.role,
    font: Number(el.dataset.fontSizePx),
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
