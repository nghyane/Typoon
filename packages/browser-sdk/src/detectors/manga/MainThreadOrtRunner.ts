import * as ort from 'onnxruntime-web/webgpu'
import type { ModelAsset } from '../../adapters/ModelAsset'
import type { CapabilityStatus, ReadyOptions, StatusListener, Unsubscribe } from '../../domain/capability'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'
import type { TextRegionRunner } from './TextRegionRunner'
import type { ComicDetrProvider, OrtWasmPaths } from './ortTypes'
import { createFeeds } from './preprocess'
import { parseDetections } from './parse'

type SessionInfo = { session: ort.InferenceSession; provider: ComicDetrProvider }

export interface MainThreadOrtRunnerOptions {
  readonly model: ModelAsset
  readonly confidenceThreshold: number
  readonly providers: readonly ComicDetrProvider[]
  readonly wasmPaths?: OrtWasmPaths
  readonly wasmNumThreads?: number
}

export class MainThreadOrtRunner implements TextRegionRunner {
  readonly name = 'manga-text-region-runner'
  private readonly failedProviders = new Set<ComicDetrProvider>()
  private readonly listeners = new Set<StatusListener>()
  private statusValue: CapabilityStatus = { name: this.name, state: 'idle' }
  private sessionPromise: Promise<SessionInfo> | null = null

  constructor(private readonly options: MainThreadOrtRunnerOptions) {
    ort.env.logLevel = 'fatal'
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths
    if (typeof options.wasmNumThreads === 'number') ort.env.wasm.numThreads = options.wasmNumThreads
  }

  status(): CapabilityStatus {
    return this.statusValue
  }

  subscribeStatus(listener: StatusListener): Unsubscribe {
    this.listeners.add(listener)
    listener(this.statusValue)
    return () => this.listeners.delete(listener)
  }

  async ensureReady(options: ReadyOptions = {}): Promise<void> {
    await this.getSession(options)
  }

  async run(image: ImagePixels, options: ReadyOptions = {}): Promise<readonly TextRegion[]> {
    const failures: string[] = []
    for (let attempt = 0; attempt < this.options.providers.length; attempt++) {
      throwIfAborted(options.signal)
      const { session, provider } = await this.getSession(options)
      try {
        throwIfAborted(options.signal)
        const output = await session.run(createFeeds(image))
        throwIfAborted(options.signal)
        return parseDetections(output, session.outputNames, image.width, image.height, this.options.confidenceThreshold)
      } catch (error) {
        if (options.signal?.aborted) throw error
        failures.push(`${provider}: ${error instanceof Error ? error.message : String(error)}`)
        this.dropProvider(session, provider, error)
      }
    }
    throw new Error(`comic-detr inference failed (${failures.join('; ')})`)
  }

  private async getSession(options: ReadyOptions): Promise<SessionInfo> {
    if (options.signal) {
      throwIfAborted(options.signal)
      if (this.statusValue.state === 'ready' && this.sessionPromise) return this.sessionPromise
      return this.createSession(options, false).then(session => {
        this.sessionPromise = Promise.resolve(session)
        return session
      })
    }
    if (!this.sessionPromise) {
      let sharedPromise: Promise<SessionInfo>
      sharedPromise = this.createSession(options, true, () => this.sessionPromise === sharedPromise)
      this.sessionPromise = sharedPromise
    }
    return this.sessionPromise
  }

  private async createSession(options: ReadyOptions, shared: boolean, isCurrent = (): boolean => true): Promise<SessionInfo> {
    try {
      await this.options.model.ensureReady(options)
      const bytes = await this.options.model.bytes(options)
      throwIfAborted(options.signal)
      this.setStatus({ name: this.name, state: 'initializing' })
      const failures: string[] = []
      for (const provider of this.options.providers) {
        if (this.failedProviders.has(provider)) continue
        if (provider === 'webgpu' && !('gpu' in navigator)) {
          failures.push('webgpu: navigator.gpu unavailable')
          continue
        }
        try {
          const session = await ort.InferenceSession.create(bytes, { executionProviders: [provider] })
          throwIfAborted(options.signal)
          console.info(`[browser-sdk] comic-detr provider=${provider}`)
          this.setStatus({ name: this.name, state: 'ready' })
          return { session, provider }
        } catch (error) {
          if (options.signal?.aborted) throw error
          failures.push(`${provider}: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
      throw new Error(`comic-detr session failed (${failures.join('; ')})`)
    } catch (error) {
      if (!options.signal?.aborted && isCurrent()) this.setStatus({ name: this.name, state: 'failed', error })
      if (shared && isCurrent()) this.sessionPromise = null
      throw error
    }
  }

  private dropProvider(session: ort.InferenceSession, provider: ComicDetrProvider, error: unknown): void {
    this.failedProviders.add(provider)
    const currentPromise = this.sessionPromise
    if (currentPromise) {
      currentPromise.then(current => {
        if (this.sessionPromise === currentPromise && current.session === session && current.provider === provider) {
          this.sessionPromise = null
        }
      }).catch(() => {})
    }
    console.warn(`[browser-sdk] comic-detr provider=${provider} failed; trying fallback`, error)
  }

  private setStatus(status: CapabilityStatus): void {
    this.statusValue = status
    for (const listener of this.listeners) listener(status)
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
