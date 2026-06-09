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

const app = document.querySelector<HTMLDivElement>('#app')!

app.innerHTML = `
  <section class="shell">
    <aside class="panel">
      <p class="eyebrow">@typoon/browser-sdk</p>
      <h1>Local overlay dev reader</h1>
      <p class="muted">
        Client-only pipeline: browser encodes the page, recognizes text with Lens, detects regions locally, then translates with Google Translate.
      </p>
      <button id="run">Run client Lens + Google Translate</button>
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
    <section class="reader">
      <div id="pageHost" class="page-host">
        <img id="page" src="/sample-page.jpg" alt="MangaDex sample page" />
      </div>
    </section>
  </section>
`

const run = document.querySelector<HTMLButtonElement>('#run')!
const status = document.querySelector<HTMLPreElement>('#status')!
const log = document.querySelector<HTMLPreElement>('#log')!
const host = document.querySelector<HTMLDivElement>('#pageHost')!
const pages = new DomPageSource(host, { imageSelector: '#page' })
const overlays = new DomOverlayTarget(host, { imageSelector: '#page' })
const erase = document.querySelector<HTMLInputElement>('#erase')!
const debugDrawable = document.querySelector<HTMLInputElement>('#debugDrawable')!
const debugTextBoxes = document.querySelector<HTMLInputElement>('#debugTextBoxes')!
const debugBounds = document.querySelector<HTMLInputElement>('#debugBounds')!
const debugLabels = document.querySelector<HTMLInputElement>('#debugLabels')!

let lastResult: TranslatedPage | null = null

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
  id: 'sample-page',
  kind: 'upload',
  pages,
  continuity: 'discrete',
})

run.addEventListener('click', async () => {
  run.disabled = true
  host.querySelector('[data-typoon-overlay="true"]')?.remove()
  log.textContent = 'running…'
  try {
    await ensureMangaFontLoaded()
    await session.ensureReady()
    lastResult = await segment.page(0).translate()
    renderLastResult()
  } catch (error) {
    log.textContent = error instanceof Error ? error.stack ?? error.message : String(error)
  } finally {
    run.disabled = false
  }
})

for (const input of [erase, debugDrawable, debugTextBoxes, debugBounds, debugLabels]) {
  input.addEventListener('change', renderLastResult)
}

function renderLastResult(): void {
  if (!lastResult) return
  host.querySelector('[data-typoon-overlay="true"]')?.remove()
  overlays.attach(0, lastResult, {
    eraseStrategy: erase.checked ? 'flat-fill' : 'none',
    debug: {
      showDrawable: debugDrawable.checked,
      showTextBoxes: debugTextBoxes.checked,
      showTextBounds: debugBounds.checked,
      showLabels: debugLabels.checked,
    },
  })
  log.textContent = JSON.stringify({
    metrics: collectTextMetrics(),
    placements: lastResult.placements.map(placement => ({
      id: placement.id,
      role: placement.role,
      text: placement.sourceText,
      bbox: placement.bbox,
      rotationDeg: placement.rotationDeg,
      textBoxes: placement.textBoxes,
      fontHint: placement.fontHint,
    })),
    translations: lastResult.translations,
  }, null, 2)
}

function collectTextMetrics(): unknown[] {
  return [...host.querySelectorAll<HTMLElement>('[data-typoon-text="true"]')].map(el => ({
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
