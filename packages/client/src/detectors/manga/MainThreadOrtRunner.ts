import type { ModelLoader } from '../../models/ModelLoader'
import { type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../../domain/capability'
import { CapabilityMachine } from '../../runtime/CapabilityMachine'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'
import type { TextRegionRunner } from './TextRegionRunner'
import type { ComicDetrProvider } from './ortTypes'
import type { OrtSessionPool } from '../../models/OrtSessionPool'
import { createFeeds } from './preprocess'
import { parseDetections } from './parse'

import type { OrtSessionHandle } from '../../models/OrtSessionPool'

export interface MainThreadOrtRunnerOptions {
  readonly model: ModelLoader
  readonly confidenceThreshold: number
  readonly providers?: readonly ComicDetrProvider[]
  readonly sessionPool: OrtSessionPool
}

export class MainThreadOrtRunner implements TextRegionRunner {
  readonly name = 'manga-text-region-runner'
  private readonly capability = new CapabilityMachine(this.name)
  private readonly options: MainThreadOrtRunnerOptions
  private sessionPromise: Promise<OrtSessionHandle> | null = null

  constructor(options: MainThreadOrtRunnerOptions) {
    this.options = options
  }

  status(): CapabilityStatus {
    return this.capability.status()
  }

  subscribeStatus(listener: StatusListener): Unsubscribe {
    return this.capability.subscribe(listener)
  }

  async ensureReady(options: ReadyOptions = {}): Promise<void> {
    await this.getSession(options)
  }

  async run(image: ImagePixels, options: ReadyOptions = {}): Promise<readonly TextRegion[]> {
    const { ort, session } = await this.getSession(options)
    throwIfAborted(options.signal)
    const output = await session.run(createFeeds(ort, image))
    throwIfAborted(options.signal)
    return parseDetections(output, session.outputNames, image.width, image.height, this.options.confidenceThreshold)
  }

  private async getSession(options: ReadyOptions): Promise<OrtSessionHandle> {
    if (options.signal) {
      throwIfAborted(options.signal)
      if (this.capability.status().state === 'ready' && this.sessionPromise) return this.sessionPromise
      return this.createSession(options)
    }
    if (!this.sessionPromise) {
      this.sessionPromise = this.createSession({})
    }
    return this.sessionPromise
  }

  private async createSession(options: ReadyOptions): Promise<OrtSessionHandle> {
    try {
      await this.options.model.ensureReady(options)
      const bytes = await this.options.model.bytes(options)
      throwIfAborted(options.signal)
      this.capability.initializing()

      const handle = await this.options.sessionPool.session(
        this.options.model.id,
        bytes,
        resolveProviders(this.options.providers, this.options.model.descriptor.executionProviders),
      )

      throwIfAborted(options.signal)
      this.capability.ready()
      return handle
    } catch (error) {
      if (!options.signal?.aborted) this.capability.failed(error)
      this.sessionPromise = null
      throw error
    }
  }
}

function resolveProviders(
  preferred: readonly ComicDetrProvider[] | undefined,
  modelProviders: readonly string[] | undefined,
): readonly ComicDetrProvider[] {
  if (preferred?.length) return preferred
  const providers = modelProviders?.filter(isComicDetrProvider) ?? []
  return providers.length ? providers : ['webgpu', 'wasm']
}

function isComicDetrProvider(provider: string): provider is ComicDetrProvider {
  return provider === 'webgpu' || provider === 'wasm'
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
