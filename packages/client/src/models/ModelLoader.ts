import { type Capability, type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../domain/capability'
import type { ModelDescriptor } from '../domain/model'
import { CapabilityMachine } from '../runtime/CapabilityMachine'
import { loadModelBytes } from '../adapters/modelFetch'
import type { ModelRegistry } from './ModelRegistry'
import type { ModelStore } from './ModelStore'

export class ModelLoader implements Capability {
  readonly id: string
  readonly name: string
  readonly descriptor: ModelDescriptor
  private readonly capability: CapabilityMachine
  private readonly store: ModelStore
  private bytesPromise: Promise<ArrayBuffer> | null = null

  constructor(
    id: string,
    registry: ModelRegistry,
    store: ModelStore,
  ) {
    this.id = id
    this.store = store
    this.descriptor = registry.get(id)
    this.name = `model:${id}`
    this.capability = new CapabilityMachine(this.name)
  }

  status(): CapabilityStatus {
    return this.capability.status()
  }

  subscribeStatus(listener: StatusListener): Unsubscribe {
    return this.capability.subscribe(listener)
  }

  async ensureReady(options: ReadyOptions = {}): Promise<void> {
    await this.bytes(options)
  }

  async bytes(options: ReadyOptions = {}): Promise<ArrayBuffer> {
    if (this.capability.status().state === 'ready' && this.bytesPromise) return this.bytesPromise

    this.bytesPromise ??= this.load(options)
    return this.bytesPromise
  }

  private async load(options: ReadyOptions): Promise<ArrayBuffer> {
    try {
      throwIfAborted(options.signal)
      this.capability.resolving()

      const cached = await this.store.read(this.descriptor)
      if (cached) {
        this.capability.ready()
        return cached
      }

      this.capability.downloading({ receivedBytes: 0, totalBytes: this.descriptor.sizeBytes, ratio: 0 })
      const bytes = await loadModelBytes(this.descriptor, progress => {
        this.capability.downloading(progress)
      }, options.signal)
      throwIfAborted(options.signal)

      // Write back to store; don't block on cache write
      this.store.write(this.descriptor, bytes).catch(() => {/* silent */})

      this.capability.ready()
      return bytes
    } catch (error) {
      if (!options.signal?.aborted) this.capability.failed(error)
      this.bytesPromise = null
      throw error
    }
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
