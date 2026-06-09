import {
  DomOverlayTarget,
  DomPageSource,
  GoogleTranslateWeb,
  LensTextRecognizer,
  MangaTextRegionDetector,
  ModelRepository,
  TranslationSession,
  type TranslatedPage,
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

const translatedPages = new Map<number, TranslatedPage>()

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

const session = new TranslationSession({
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
})

session.subscribeStatus(snapshot => {
  status.textContent = JSON.stringify({ status: snapshot.capabilities }, null, 2)
})

const work = session.openWork({ id: 'dev-sample', sourceLang: 'en', targetLang: 'vi' })
const segment = work.openSegment({
  id: 'chapter-20',
  kind: 'chapter',
  pages,
  continuity: 'continuous',
})

run.addEventListener('click', async () => {
  run.disabled = true
  clearOverlays()
  translatedPages.clear()
  log.textContent = 'running chapter…'
  try {
    await ensureMangaFontLoaded()
    await session.ensureReady()
    const startedAt = performance.now()
    for (let index = 0; index < segment.pageCount; index++) {
      logProgress(index, startedAt)
      const page = await segment.page(index).translate()
      translatedPages.set(index, page)
      overlays.attach(index, page, overlayOptions())
      logProgress(index + 1, startedAt)
    }
  } catch (error) {
    log.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  } finally {
    run.disabled = false
  }
})

for (const input of [erase, debugDrawable, debugTextBoxes, debugBounds, debugLabels]) {
  input.addEventListener('change', renderTranslatedPages)
}

function renderTranslatedPages(): void {
  clearOverlays()
  for (const [index, page] of translatedPages) overlays.attach(index, page, overlayOptions())
  logProgress(translatedPages.size, null)
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

function logProgress(done: number, startedAt: number | null): void {
  const elapsedMs = startedAt === null ? null : Math.round(performance.now() - startedAt)
  log.textContent = JSON.stringify({
    progress: {
      done,
      total: segment.pageCount,
      elapsedMs,
    },
    metrics: collectTextMetrics(),
    pages: [...translatedPages].map(([pageIndex, page]) => ({
      pageIndex,
      placementCount: page.placements.length,
      translationCount: page.translations.length,
      placements: page.placements.map(placement => ({
        id: placement.id,
        role: placement.role,
        text: placement.sourceText,
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
