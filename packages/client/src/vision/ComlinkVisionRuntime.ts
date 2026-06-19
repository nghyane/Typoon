/**
 * Vision runtime that delegates all work to a Web Worker via Comlink.
 *
 * The worker exposes a {@link WorkerVisionImpl} instance.  Comlink
 * proxies method calls transparently — no postMessage, no requestId
 * correlation, no switch/case dispatch.
 *
 * ONNX detection (TextRegionDetector) still runs on the main thread
 * because onnxruntime-web requires WebGL / WebGPU.  This runtime
 * reads pixels back for detection, then calls the detector locally.
 */
import * as Comlink from 'comlink'
import type { ImagePixels } from '../domain/image'
import type { PreparedPageHandle } from '../domain/prepared'
import type { PageAsset } from '../domain/source'
import type { TextRegion } from '../domain/regions'
import type { PreparationStrategy } from '../domain/run'
import type { TextPlacement } from '../domain/planning'
import type { SafeMarginsDebug } from '../render/backgroundFit'
import type { EncodedOcrImage } from '../recognizers/text'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { PreparationSession, VisionRuntime } from './VisionRuntime'
import type { WorkerVisionImpl } from './WorkerVisionImpl'

type WorkerProxy = Comlink.Remote<WorkerVisionImpl>

export class ComlinkVisionRuntime implements VisionRuntime {
  private readonly proxy: WorkerProxy
  private readonly worker: Worker
  private readonly detector: TextRegionDetector | null
  private workerError: Error | null = null
  private disposed = false

  constructor(worker: Worker, options: {
    readonly detector?: TextRegionDetector
  } = {}) {
    this.worker = worker
    this.detector = options.detector ?? null
    this.proxy = Comlink.wrap<WorkerVisionImpl>(worker)
    worker.addEventListener('error', event => {
      this.workerError = workerEventError(event, 'vision worker failed')
    })
    worker.addEventListener('messageerror', () => {
      this.workerError = new Error('vision worker message failed')
    })
  }

  async beginPreparation(
    runId: string,
    strategy: PreparationStrategy,
  ): Promise<PreparationSession> {
    await this.callWorker(this.proxy.beginPreparation(runId, strategy), 'beginPreparation')
    return {
      push: (asset: PageAsset, _signal?: AbortSignal) => this.callWorker(this.proxy.pushPreparation(runId, asset), 'pushPreparation', 30_000),
      flush: (_signal?: AbortSignal) => this.callWorker(this.proxy.flushPreparation(runId), 'flushPreparation', 30_000),
      dispose: () => { void this.proxy.disposePreparation(runId) },
    }
  }

  async readPixels(
    handle: PreparedPageHandle,
    _signal?: AbortSignal,
  ): Promise<ImagePixels> {
    return this.callWorker(this.proxy.readPixels(handle.runId, handle.preparedPageId), 'readPixels', 30_000)
  }

  async encodeForOcr(
    handle: PreparedPageHandle,
    _signal?: AbortSignal,
  ): Promise<EncodedOcrImage> {
    return this.callWorker(this.proxy.encodeForOcr(handle.runId, handle.preparedPageId), 'encodeForOcr', 30_000)
  }

  async estimateMargins(
    handle: PreparedPageHandle,
    placements: readonly TextPlacement[],
    _signal?: AbortSignal,
  ): Promise<readonly SafeMarginsDebug[]> {
    return this.callWorker(this.proxy.estimateMargins(handle.runId, handle.preparedPageId, placements), 'estimateMargins', 30_000)
  }

  async createSeamRepair(
    top: PreparedPageHandle,
    bottom: PreparedPageHandle,
    bandPx: number,
    _signal?: AbortSignal,
  ): Promise<PreparedPageHandle | null> {
    return this.callWorker(this.proxy.createSeamRepair(top.runId, top.preparedPageId, bottom.preparedPageId, bandPx), 'createSeamRepair', 30_000)
  }

  async detectTextRegions(
    handle: PreparedPageHandle,
    signal?: AbortSignal,
  ): Promise<readonly TextRegion[]> {
    if (!this.detector) return []
    const image = await this.readPixels(handle, signal)
    return this.detector.detectTextRegions(image, { signal })
  }

  release(handle: PreparedPageHandle): void {
    void this.proxy.release(handle.runId, handle.preparedPageId)
  }

  cancelRun(runId: string): void {
    void this.proxy.cancelRun(runId)
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    void this.proxy.dispose()
    ;(this.proxy as any)[Comlink.releaseProxy]()
    this.worker.terminate()
  }

  private callWorker<T>(promise: Promise<T>, label: string, timeoutMs = 10_000): Promise<T> {
    if (this.workerError) return Promise.reject(this.workerError)
    if (this.disposed) return Promise.reject(new Error('vision worker is disposed'))

    return new Promise<T>((resolve, reject) => {
      let settled = false
      const finish = (fn: () => void): void => {
        if (settled) return
        settled = true
        cleanup()
        fn()
      }
      const onError = (event: ErrorEvent): void => {
        const error = workerEventError(event, 'vision worker failed')
        this.workerError = error
        finish(() => reject(error))
      }
      const onMessageError = (): void => {
        const error = new Error('vision worker message failed')
        this.workerError = error
        finish(() => reject(error))
      }
      const timer = setTimeout(() => {
        finish(() => reject(new Error(`vision worker ${label} timed out`)))
      }, timeoutMs)
      const cleanup = (): void => {
        clearTimeout(timer)
        this.worker.removeEventListener('error', onError)
        this.worker.removeEventListener('messageerror', onMessageError)
      }

      this.worker.addEventListener('error', onError, { once: true })
      this.worker.addEventListener('messageerror', onMessageError, { once: true })
      promise.then(
        value => finish(() => resolve(value)),
        error => finish(() => reject(error instanceof Error ? error : new Error(String(error)))),
      )
    })
  }
}

function workerEventError(event: ErrorEvent, fallback: string): Error {
  if (event.error instanceof Error) return event.error
  const message = event.message || fallback
  const parts = [message]
  if (event.filename) parts.push(event.filename)
  if (event.lineno) parts.push(`:${event.lineno}${event.colno ? `:${event.colno}` : ''}`)
  return new Error(parts.join(' '))
}
