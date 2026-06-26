// reader/visionRuntime.ts — lazy singleton for OCR region-detection model + ORT.
// Extracted from the monolithic translation controller so model state and
// runtime wiring live in one place.

import { detectBrowserCapabilities, type BrowserCapabilities } from '../adapters/browserCapabilities'
import { ModelRepository } from '../adapters/ModelRepository'
import { proxyUrl } from '../adapters/huggingFace'
import { MangaTextRegionDetector } from '../detectors/manga/MangaTextRegionDetector'
import { MainThreadOrtRunner } from '../detectors/manga/MainThreadOrtRunner'
import { WorkerOrtRunner } from '../detectors/manga/WorkerOrtRunner'
import type { TextRegionRunner } from '../detectors/manga/TextRegionRunner'
import { COMIC_DETR_DEFAULT_CONFIDENCE } from '../detectors/manga/ortTypes'
import type { ModelLoader } from '../models/ModelLoader'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { CapabilityStatus } from '../domain/capability'
import type { ImagePixels } from '../domain/image'
import type { TextRegion } from '../domain/regions'
import { OrtRuntime } from '../models/OrtRuntime'
import { OrtSessionPool, type OrtProvider } from '../models/OrtSessionPool'
import type { OrtModule } from '../models/OrtBackend'
import { defaultTranslationConfig, preferredProviders, type TranslationConfig } from './translationConfig'
import { errorMessage, neverAbort, throwIfAborted, whenIdle } from './asyncSignal'
import ortWebgpuMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url'
import ortWebgpuWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url'
import ortWasmMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'

export interface ReaderModelState {
  readonly state: 'idle' | 'resolving' | 'downloading' | 'initializing' | 'ready' | 'failed'
  readonly receivedBytes?: number
  readonly totalBytes?: number
  readonly ratio?: number
  readonly error?: string
}

interface ConfiguredOrtRuntime {
  readonly sessionPool: OrtSessionPool
  readonly providers: readonly OrtProvider[]
}

interface OrtBackend {
  readonly ort: OrtModule
  readonly wasmPaths: { readonly wasm: string; readonly mjs: string }
}

let modelRepository: ModelRepository | null = null
let ortRuntimePromise: Promise<ConfiguredOrtRuntime> | null = null
let detectorPromise: Promise<TextRegionDetector> | null = null
let latestModelState: ReaderModelState = { state: 'idle' }
const modelStateListeners = new Set<(state: ReaderModelState) => void>()

function repository(config: TranslationConfig): ModelRepository {
  // Config-first-wins is intentional: the app uses a single TranslationConfig
  // for its lifetime, so a later differing config never reaches here. If
  // per-config repositories are ever needed, key this by repo/revision instead.
  if (!modelRepository) {
    modelRepository = new ModelRepository({
      manifestUrl: proxyUrl(config.model.manifestUrl, config.model.proxyBase),
      resolveUrl: url => proxyUrl(url, config.model.proxyBase),
    })
  }
  return modelRepository
}

export function subscribeModelState(listener: (state: ReaderModelState) => void): () => void {
  modelStateListeners.add(listener)
  listener(latestModelState)
  return () => modelStateListeners.delete(listener)
}

export async function detectTextRegions(
  image: ImagePixels,
  signal: AbortSignal,
  config: TranslationConfig = defaultTranslationConfig,
): Promise<readonly TextRegion[]> {
  const detector = await defaultTextRegionDetector(signal, config)
  return detector.detectTextRegions(image, { signal })
}

/**
 * Warm the detector singleton in the background so the first OCR page does not
 * pay for the ORT backend import + model download + session compile. Safe to
 * call repeatedly; a no-op once a load is in flight or done. Scheduled on idle
 * so it never competes with the reader's first paint.
 */
export function prewarmTextRegionDetector(config: TranslationConfig = defaultTranslationConfig): void {
  if (detectorPromise) return
  whenIdle(() => {
    if (detectorPromise) return
    // neverAbort: prewarm load should outlive a cancelled translate. Errors are
    // swallowed here (state is published via model listeners) and detectorPromise
    // self-resets on failure so a real run retries cleanly.
    void defaultTextRegionDetector(neverAbort(), config).catch(() => {})
  })
}

function defaultTextRegionDetector(signal: AbortSignal, config: TranslationConfig): Promise<TextRegionDetector> {
  return lazyOnce(
    () => detectorPromise,
    p => { detectorPromise = p },
    () => createTextRegionDetector(signal, config),
  )
}

async function createTextRegionDetector(signal: AbortSignal, config: TranslationConfig): Promise<TextRegionDetector> {
  throwIfAborted(signal)
  const caps = detectBrowserCapabilities()
  const model = await repository(config).model(caps.modelHint.modelId)

  if (config.detector.useWorker) {
    try {
      return await startDetector(model, workerRunner(caps, model), signal)
    } catch (error) {
      if (signal.aborted) throw error
      // Worker path failed (build/runtime/provider): fall back to main thread so
      // detection still works; the model bytes are already resolved.
    }
  }
  return startDetector(model, await mainThreadRunner(caps, model), signal)
}

async function startDetector(model: ModelLoader, runner: TextRegionRunner, signal: AbortSignal): Promise<TextRegionDetector> {
  const detector = new MangaTextRegionDetector({ model, runner })
  detector.subscribeStatus(status => publishModelState(capabilityToModelState(status)))
  publishModelState(capabilityToModelState(detector.status()))
  await detector.ensureReady({ signal })
  return detector
}

async function mainThreadRunner(caps: BrowserCapabilities, model: ModelLoader): Promise<TextRegionRunner> {
  const ortRuntime = await configureOrtRuntime(caps)
  return new MainThreadOrtRunner({
    model,
    confidenceThreshold: COMIC_DETR_DEFAULT_CONFIDENCE,
    providers: ortRuntime.providers,
    sessionPool: ortRuntime.sessionPool,
  })
}

function workerRunner(caps: BrowserCapabilities, model: ModelLoader): TextRegionRunner {
  const provider = caps.modelHint.preferredProvider
  return new WorkerOrtRunner({
    model,
    confidenceThreshold: COMIC_DETR_DEFAULT_CONFIDENCE,
    provider,
    wasmPaths: resolveWasmPaths(provider),
    numThreads: caps.modelHint.wasmNumThreads,
  })
}

function resolveWasmPaths(provider: OrtProvider): { wasm: string; mjs: string } {
  return provider === 'webgpu'
    ? absoluteWasmPaths(ortWebgpuWasmUrl, ortWebgpuMjsUrl)
    : absoluteWasmPaths(ortWasmUrl, ortWasmMjsUrl)
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
  return lazyOnce(
    () => ortRuntimePromise,
    p => { ortRuntimePromise = p },
    () => createOrtRuntime(caps),
  )
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

/**
 * Memoize an async factory in a caller-owned slot: run it once, cache the
 * promise, and reset the slot on rejection so a later call retries cleanly.
 */
function lazyOnce<T>(
  get: () => Promise<T> | null,
  set: (promise: Promise<T> | null) => void,
  factory: () => Promise<T>,
): Promise<T> {
  const existing = get()
  if (existing) return existing
  const promise = factory().catch(error => {
    set(null)
    throw error
  })
  set(promise)
  return promise
}
