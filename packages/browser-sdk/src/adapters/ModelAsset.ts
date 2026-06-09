import type { Capability, CapabilityStatus, ReadyOptions, StatusListener, Unsubscribe } from '../domain/capability'
import { BrowserModelCache } from './BrowserModelCache'
import { loadModelBytes } from './modelFetch'
import type { ModelDescriptor, ModelManifest, ModelRepositoryOptions, StoredModel } from './modelTypes'

export class ModelAsset implements Capability {
  readonly name: string
  private statusValue: CapabilityStatus
  private readonly listeners = new Set<StatusListener>()
  private storedPromise: Promise<StoredModel> | null = null

  constructor(
    readonly id: string,
    private readonly loadManifest: (signal?: AbortSignal) => Promise<ModelManifest>,
    private readonly options: ModelRepositoryOptions,
  ) {
    this.name = `model:${id}`
    this.statusValue = { name: this.name, state: 'idle' }
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
    await this.stored(options)
  }

  async bytes(options: ReadyOptions = {}): Promise<ArrayBuffer> {
    return (await this.stored(options)).bytes
  }

  async descriptor(options: ReadyOptions = {}): Promise<ModelDescriptor> {
    return (await this.stored(options)).descriptor
  }

  private stored(options: ReadyOptions): Promise<StoredModel> {
    if (options.signal) {
      throwIfAborted(options.signal)
      if (this.statusValue.state === 'ready' && this.storedPromise) return this.storedPromise
      return this.load(options, false).then(stored => {
        this.storedPromise = Promise.resolve(stored)
        return stored
      })
    }
    if (!this.storedPromise || this.statusValue.state === 'failed') {
      let sharedPromise: Promise<StoredModel>
      sharedPromise = this.load(options, true, () => this.storedPromise === sharedPromise)
      this.storedPromise = sharedPromise
    }
    return this.storedPromise
  }

  private async load(options: ReadyOptions, shared: boolean, isCurrent = (): boolean => true): Promise<StoredModel> {
    try {
      throwIfAborted(options.signal)
      this.setStatus({ name: this.name, state: 'resolving' })
      const manifest = await this.loadManifest(options.signal)
      throwIfAborted(options.signal)
      const descriptor = manifest.models[this.id]
      if (!descriptor) throw new Error(`model not found in manifest: ${this.id}`)
      const cache = this.options.cache ?? new BrowserModelCache(this.options.cacheName ?? `typoon-models-${manifest.version}`)
      const bytes = await loadModelBytes(descriptor, cache, status => this.setStatus(status), this.name, options.signal)
      throwIfAborted(options.signal)
      const stored = { id: this.id, descriptor, bytes }
      this.setStatus({ name: this.name, state: 'ready' })
      return stored
    } catch (error) {
      if (!options.signal?.aborted && isCurrent()) this.setStatus({ name: this.name, state: 'failed', error })
      if (shared && isCurrent()) this.storedPromise = null
      throw error
    }
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
