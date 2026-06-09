import {
  DomOverlayTarget,
  DomPageSource,
  GoogleTranslateWeb,
  LensTextRecognizer,
  MangaTextRegionDetector,
  ModelRepository,
  TranslationEngine,
  type RenderedPage,
  ensureMangaFontLoaded,
} from '../../src/index'
import ortWebgpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'
import './style.css'

const comicDetrModelUrl = new URL('../../../../workers/scan/container/comic-detr-v4s-int8.onnx', import.meta.url).href
const ortWasmUrl = new URL(ortWebgpuWasmUrl, window.location.href).href
const chapterPages = Array.from({ length: 20 }, (_, index) => `/chapter-20/${String(index + 1).padStart(3, '0')}.jpg`)

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <section class="shell">
    <aside class="panel">
      <p class="eyebrow">@typoon/browser-sdk</p>
      <h1>Local overlay dev reader</h1>
      <p class="muted">
        Client-only pipeline: browser encodes the page, recognizes text with Lens, detects regions locally, then translates with Google Translate.
      </p>
      <button id="run">Translate 20-page chapter</button>
      <div class="controls" aria-label="overlay debug controls">
        <label><input id="erase" type="checkbox" checked /> erase flat-fill</label>
        <label><input id="debugDrawable" type="checkbox" /> drawable</label>
        <label><input id="debugTextBoxes" type="checkbox" /> OCR boxes</label>
        <label><input id="debugBounds" type="checkbox" /> plan bounds</label>
        <label><input id="debugLabels" type="checkbox" /> labels</label>
      </div>
      <pre id="status">status idle</pre>
      <pre id="log">idle</pre>
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
const reader = document.querySelector<HTMLDivElement>('#reader')!
const pages = new DomPageSource(reader, { imageSelector: '.page-image' })
const overlays = new DomOverlayTarget(reader, { imageSelector: '.page-image', hostSelector: '.page-host' })
const erase = document.querySelector<HTMLInputElement>('#erase')!
const debugDrawable = document.querySelector<HTMLInputElement>('#debugDrawable')!
const debugTextBoxes = document.querySelector<HTMLInputElement>('#debugTextBoxes')!
const debugBounds = document.querySelector<HTMLInputElement>('#debugBounds')!
const debugLabels = document.querySelector<HTMLInputElement>('#debugLabels')!

const renderedPages = new Map<number, RenderedPage>()

const models = new ModelRepository({
  manifest: {
    version: 'dev-local',
    models: {
      comicDetr: {
        url: comicDetrModelUrl,
        inputSize: 640,
        executionProviders: ['webgpu', 'wasm'],
      },
    },
  },
})

const engine = new TranslationEngine({
  sourceLang: 'en',
  targetLang: 'vi',
  recognizer: new LensTextRecognizer(),
  detector: new MangaTextRegionDetector({
    model: models.model('comicDetr'),
    preferredProviders: ['webgpu', 'wasm'],
    wasmPaths: { wasm: ortWasmUrl },
    wasmNumThreads: 1,
  }),
  translator: new GoogleTranslateWeb(),
  scheduler: {
    concurrency: { pages: 4, recognize: 4, detect: 1, translate: 4 },
    retry: { recognize: { attempts: 2 }, translate: { attempts: 2 } },
  },
})

status.textContent = 'status idle'

run.addEventListener('click', async () => {
  run.disabled = true
  clearOverlays()
  clearPendingOverlays()
  clearFailedOverlays()
  renderedPages.clear()
  log.textContent = 'running chapter…'
  try {
    await ensureMangaFontLoaded()
    const startedAt = performance.now()
    markPendingPages()
    const segmentRun = engine.translateSegment({
      workId: 'dev-sample',
      segmentId: 'chapter-20',
      source: pages,
      sourceLang: 'en',
      targetLang: 'vi',
      stopOnError: false,
    })
    for await (const event of segmentRun.events()) {
      status.textContent = `event ${event.type}`
      if (event.type === 'page-display-ready') {
        renderedPages.set(event.pageIndex, event.page)
        setPagePending(event.pageIndex, false)
        overlays.attach(event.pageIndex, event.page, overlayOptions())
        logProgress(renderedPages.size, startedAt)
      } else if (event.type === 'page-layout-failed') {
        logProgress(renderedPages.size, startedAt)
      } else if (event.type === 'page-failed') {
        setPagePending(event.pageIndex, false)
        setPageFailed(event.pageIndex, event.error)
        logProgress(renderedPages.size, startedAt)
      }
    }
    await segmentRun.done()
  } catch (error) {
    log.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  } finally {
    run.disabled = false
  }
})

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

function setPageFailed(index: number, error: unknown): void {
  const host = pageHost(index)
  host.querySelector('[data-dev-failed-overlay="true"]')?.remove()
  const overlay = document.createElement('div')
  overlay.dataset.devFailedOverlay = 'true'
  const message = error instanceof Error ? error.message : String(error)
  overlay.textContent = `Failed: ${message}`
  host.appendChild(overlay)
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
