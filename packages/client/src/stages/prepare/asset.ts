import type { ImagePixels } from '../../domain/image'
import type { PreparedPageAsset } from '../../domain/preparedChapter'
import type { CanvasBackend } from './canvasBackend'

export class MemoryPreparedPageAsset implements PreparedPageAsset {
  private pixels: ImagePixels | null
  private blobPromise: Promise<Blob> | null = null
  private readonly backend: CanvasBackend

  constructor(image: ImagePixels, backend: CanvasBackend) {
    this.pixels = image
    this.backend = backend
  }

  async readPixels(signal?: AbortSignal): Promise<ImagePixels> {
    if (signal?.aborted) throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
    if (!this.pixels) throw new Error('prepared pixels released')
    return this.pixels
  }

  async readBlob(signal?: AbortSignal): Promise<Blob> {
    this.blobPromise ??= this.backend.encodePng(await this.readPixels(signal), signal)
    return this.blobPromise
  }

  release(): void {
    this.pixels = null
  }
}
