// reader/visionRuntime.ts — lazy singleton for OCR region-detection model + ORT.
// Extracted from the monolithic translation controller so model state and
// runtime wiring live in one place.

import { detectBrowserCapabilities, type BrowserCapabilities } from '../adapters/browserCapabilities'
import { ModelRepository } from '../adapters/ModelRepository'
import { MangaTextRegionDetector } from '../detectors/manga/MangaTextRegionDetector'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { CapabilityStatus } from '../domain/capability'
import type { ImagePixels } from '../domain/image'
import type { TextRegion } from '../domain/regions'
import { OrtRuntime } from '../models/OrtRuntime'
import { OrtSessionPool, type OrtProvider } from '../models/OrtSessionPool'
import type { OrtModule } from '../models/OrtBackend'
import { defaultTranslationConfig, preferredProviders, type TranslationConfig } from './translationConfig'
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
  if (!modelRepository) {
    modelRepository = ModelRepository.fromHuggingFace({
      repo: config.model.repo,
      revision: config.model.revision,
      proxyBase: config.model.proxyBase,
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
    // Never-aborting signal: prewarm load should outlive a cancelled translate.
    // Errors are swallowed here (state is published via model listeners) and
    // detectorPromise self-resets on failure so a real run retries cleanly.
    void defaultTextRegionDetector(new AbortController().signal, config).catch(() => {})
  })
}

function defaultTextRegionDetector(signal: AbortSignal, config: TranslationConfig): Promise<TextRegionDetector> {
  if (!detectorPromise) {
    detectorPromise = createTextRegionDetector(signal, config).catch(error => {
      detectorPromise = null
      throw error
    })
  }
  return detectorPromise
}

async function createTextRegionDetector(signal: AbortSignal, config: TranslationConfig): Promise<TextRegionDetector> {
  throwIfAborted(signal)
  const caps = detectBrowserCapabilities()
  const ortRuntime = await configureOrtRuntime(caps)
  const model = await repository(config).model(caps.modelHint.modelId)
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

function throwIfAborted(signal: AbortSignal): void {
  if (!signal.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

function whenIdle(run: () => void): void {
  const win = globalThis as typeof globalThis & {
    requestIdleCallback?: (cb: () => void, opts?: { timeout?: number }) => number
  }
  if (typeof win.requestIdleCallback === 'function') win.requestIdleCallback(run, { timeout: 2000 })
  else setTimeout(run, 0)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
