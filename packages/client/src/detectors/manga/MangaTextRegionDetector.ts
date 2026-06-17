import type { ModelLoader } from '../../models/ModelLoader'
import { type Capability, type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../../domain/capability'
import { CapabilityMachine } from '../../runtime/CapabilityMachine'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'
import type { TextRegionDetector } from '../textRegions'
import { MainThreadOrtRunner } from './MainThreadOrtRunner'
import { COMIC_DETR_DEFAULT_CONFIDENCE, type ComicDetrProvider } from './ortTypes'
import type { TextRegionRunner } from './TextRegionRunner'
import type { OrtSessionPool } from '../../models/OrtSessionPool'

export interface ComicDetrDetectorOptions {
  readonly model: ModelLoader
  readonly confidenceThreshold?: number
  readonly preferredProviders?: readonly ComicDetrProvider[]
  readonly sessionPool: OrtSessionPool
}

export class MangaTextRegionDetector implements TextRegionDetector, Capability {
  readonly name = 'manga-text-region-detector'
  private readonly capability = new CapabilityMachine(this.name)
  private readonly runner: TextRegionRunner

  constructor(options: ComicDetrDetectorOptions) {
    this.runner = new MainThreadOrtRunner({
      model: options.model,
      confidenceThreshold: options.confidenceThreshold ?? COMIC_DETR_DEFAULT_CONFIDENCE,
      providers: options.preferredProviders,
      sessionPool: options.sessionPool,
    })
    options.model.subscribeStatus(status => {
      if (status.state === 'resolving' || status.state === 'downloading' || status.state === 'failed') {
        this.capability.mirror(status)
      }
    })
    this.runner.subscribeStatus(status => {
      if (status.state === 'initializing' || status.state === 'ready' || status.state === 'failed') {
        this.capability.mirror(status)
      }
    })
  }

  status(): CapabilityStatus {
    return this.capability.status()
  }

  subscribeStatus(listener: StatusListener): Unsubscribe {
    return this.capability.subscribe(listener)
  }

  async ensureReady(options: ReadyOptions = {}): Promise<void> {
    await this.runner.ensureReady(options)
  }

  detectTextRegions(image: ImagePixels, options?: ReadyOptions): Promise<readonly TextRegion[]> {
    return this.runner.run(image, options)
  }

}
