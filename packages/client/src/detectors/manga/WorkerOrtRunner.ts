import * as Comlink from 'comlink'
import type { ModelLoader } from '../../models/ModelLoader'
import { type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../../domain/capability'
import { CapabilityMachine } from '../../runtime/CapabilityMachine'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'
import type { Tensor } from 'onnxruntime-web/wasm'
import type { TextRegionRunner } from './TextRegionRunner'
import type { ComicDetrProvider } from './ortTypes'
import { prepareImageTensor } from './preprocess'
import { parseDetections } from './parse'
import type { OrtWorkerApi } from './ortWorkerProtocol'

export interface WorkerOrtRunnerOptions {
  readonly model: ModelLoader
  readonly confidenceThreshold: number
  readonly provider: ComicDetrProvider
  readonly wasmPaths: { readonly wasm: string; readonly mjs: string }
  readonly numThreads: number
}

/**
 * Runs Comic-DETR inference in a Web Worker. Preprocessing stays on the main
 * thread (cheap canvas resize); only the planar tensor and raw output tensors
 * cross the boundary, so the blocking session.run() never janks the UI.
 */
export class WorkerOrtRunner implements TextRegionRunner {
  readonly name = 'manga-text-region-worker-runner'
  private readonly capability = new CapabilityMachine(this.name)
  private worker: Worker | null = null
  private readyPromise: Promise<Comlink.Remote<OrtWorkerApi>> | null = null

  constructor(private readonly options: WorkerOrtRunnerOptions) {}

  status(): CapabilityStatus {
    return this.capability.status()
  }

  subscribeStatus(listener: StatusListener): Unsubscribe {
    return this.capability.subscribe(listener)
  }

  async ensureReady(options: ReadyOptions = {}): Promise<void> {
    await this.init(options)
  }

  async run(image: ImagePixels, options: ReadyOptions = {}): Promise<readonly TextRegion[]> {
    const remote = await this.init(options)
    throwIfAborted(options.signal)
    const images = prepareImageTensor(image)
    const result = await remote.run(
      Comlink.transfer({ images, width: image.width, height: image.height }, [images.buffer]),
    )
    throwIfAborted(options.signal)
    const outputs = result.outputs as unknown as Record<string, Tensor>
    return parseDetections(outputs, result.outputNames, image.width, image.height, this.options.confidenceThreshold)
  }

  private init(options: ReadyOptions): Promise<Comlink.Remote<OrtWorkerApi>> {
    if (options.signal) throwIfAborted(options.signal)
    this.readyPromise ??= this.create(options)
    return this.readyPromise
  }

  private async create(options: ReadyOptions): Promise<Comlink.Remote<OrtWorkerApi>> {
    try {
      this.capability.initializing()
      const bytes = await this.options.model.bytes(options)
      throwIfAborted(options.signal)
      const worker = new Worker(new URL('./ortWorker.ts', import.meta.url), { type: 'module' })
      const remote = Comlink.wrap<OrtWorkerApi>(worker)
      // Clone (not transfer) the model bytes: model.bytes() caches the same
      // ArrayBuffer, so detaching it would break a later re-init.
      await remote.init({
        bytes,
        provider: this.options.provider,
        wasmPaths: this.options.wasmPaths,
        numThreads: this.options.numThreads,
      })
      this.worker = worker
      this.capability.ready()
      return remote
    } catch (error) {
      this.readyPromise = null
      this.worker?.terminate()
      this.worker = null
      if (!options.signal?.aborted) this.capability.failed(error)
      throw error
    }
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
