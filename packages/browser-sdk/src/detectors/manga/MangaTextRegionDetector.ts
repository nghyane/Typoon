import type { ModelAsset } from '../../adapters/ModelAsset'
import type { Capability, CapabilityStatus, ReadyOptions, StatusListener, Unsubscribe } from '../../domain/capability'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'
import type { TextRegionDetector } from '../textRegions'
import { MainThreadOrtRunner } from './MainThreadOrtRunner'
import { COMIC_DETR_DEFAULT_CONFIDENCE, type ComicDetrProvider, type OrtWasmPaths } from './ortTypes'
import type { TextRegionRunner } from './TextRegionRunner'

export interface ComicDetrDetectorOptions {
  readonly model: ModelAsset
  readonly confidenceThreshold?: number
  readonly preferredProviders?: readonly ComicDetrProvider[]
  readonly wasmPaths?: OrtWasmPaths
  readonly wasmNumThreads?: number
}

export class MangaTextRegionDetector implements TextRegionDetector, Capability {
  readonly name = 'manga-text-region-detector'
  private readonly listeners = new Set<StatusListener>()
  private readonly runner: TextRegionRunner
  private statusValue: CapabilityStatus = { name: this.name, state: 'idle' }

  constructor(options: ComicDetrDetectorOptions) {
    this.runner = new MainThreadOrtRunner({
      model: options.model,
      confidenceThreshold: options.confidenceThreshold ?? COMIC_DETR_DEFAULT_CONFIDENCE,
      providers: options.preferredProviders ?? ['webgpu', 'wasm'],
      wasmPaths: options.wasmPaths,
      wasmNumThreads: options.wasmNumThreads,
    })
    options.model.subscribeStatus(status => {
      if (status.state === 'resolving' || status.state === 'downloading' || status.state === 'failed') {
        this.setStatus({ name: this.name, state: status.state, progress: status.progress, error: status.error })
      }
    })
    this.runner.subscribeStatus(status => {
      if (status.state === 'initializing' || status.state === 'ready' || status.state === 'failed') {
        this.setStatus({ name: this.name, state: status.state, progress: status.progress, error: status.error })
      }
    })
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
    await this.runner.ensureReady(options)
  }

  detectTextRegions(image: ImagePixels, options?: ReadyOptions): Promise<readonly TextRegion[]> {
    return this.runner.run(image, options)
  }

  private setStatus(status: CapabilityStatus): void {
    this.statusValue = status
    for (const listener of this.listeners) listener(status)
  }
}
