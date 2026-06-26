import type { ModelLoader } from '../../models/ModelLoader'
import { type Capability, type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../../domain/capability'
import { CapabilityMachine } from '../../runtime/CapabilityMachine'
import type { ImagePixels } from '../../domain/image'
import type { TextRegion } from '../../domain/regions'
import type { TextRegionDetector } from '../textRegions'
import type { TextRegionRunner } from './TextRegionRunner'

export interface ComicDetrDetectorOptions {
  readonly model: ModelLoader
  /** Injected inference backend (main-thread ORT or worker ORT). */
  readonly runner: TextRegionRunner
}

export class MangaTextRegionDetector implements TextRegionDetector, Capability {
  readonly name = 'manga-text-region-detector'
  private readonly capability = new CapabilityMachine(this.name)
  private readonly runner: TextRegionRunner

  constructor(options: ComicDetrDetectorOptions) {
    this.runner = options.runner
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

  async detectTextRegions(image: ImagePixels, options?: ReadyOptions): Promise<readonly TextRegion[]> {
    await this.ensureReady(options)
    return this.runner.run(image, options)
  }

}
