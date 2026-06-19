import type { ImagePixels } from '../../domain/image'

export interface DownsampleBandArgs {
  readonly image: ImagePixels
  readonly edge: 'top' | 'bottom'
  readonly bandPx: number
  readonly targetWidthPx: number
  readonly signal?: AbortSignal
}

export interface CanvasBackend {
  decode(blob: Blob, signal?: AbortSignal): Promise<ImagePixels>
  stitchVertical(images: readonly ImagePixels[], signal?: AbortSignal): Promise<ImagePixels>
  encodePng(image: ImagePixels, signal?: AbortSignal): Promise<Blob>
  downsampleBand(args: DownsampleBandArgs): Promise<ImagePixels>
}

export class BrowserCanvasBackend implements CanvasBackend {
  async decode(blob: Blob, signal?: AbortSignal): Promise<ImagePixels> {
    throwIfAborted(signal)
    const bitmap = await createImageBitmap(blob)
    try {
      throwIfAborted(signal)
      const canvas = createCanvas(bitmap.width, bitmap.height)
      const ctx = context2d(canvas)
      ctx.drawImage(bitmap, 0, 0)
      const image = ctx.getImageData(0, 0, bitmap.width, bitmap.height)
      return { width: image.width, height: image.height, data: image.data }
    } finally {
      bitmap.close()
    }
  }

  async stitchVertical(images: readonly ImagePixels[], signal?: AbortSignal): Promise<ImagePixels> {
    throwIfAborted(signal)
    if (!images.length) throw new Error('cannot stitch empty image list')
    const width = Math.max(...images.map(image => image.width))
    const height = images.reduce((sum, image) => sum + image.height, 0)
    const canvas = createCanvas(width, height)
    const ctx = context2d(canvas)
    let y = 0
    for (const image of images) {
      throwIfAborted(signal)
      ctx.putImageData(new ImageData(image.data, image.width, image.height), 0, y)
      y += image.height
    }
    const stitched = ctx.getImageData(0, 0, width, height)
    return { width: stitched.width, height: stitched.height, data: stitched.data }
  }

  async encodePng(image: ImagePixels, signal?: AbortSignal): Promise<Blob> {
    throwIfAborted(signal)
    const canvas = createCanvas(image.width, image.height)
    const ctx = context2d(canvas)
    ctx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)
    if ('convertToBlob' in canvas) {
      return (canvas as OffscreenCanvas).convertToBlob({ type: 'image/png' })
    }
    return new Promise((resolve, reject) => {
      ;(canvas as HTMLCanvasElement).toBlob(blob => blob ? resolve(blob) : reject(new Error('failed to encode prepared image')), 'image/png')
    })
  }

  async downsampleBand(args: DownsampleBandArgs): Promise<ImagePixels> {
    throwIfAborted(args.signal)
    const sourceH = Math.min(args.bandPx, args.image.height)
    const sourceY = args.edge === 'top' ? 0 : Math.max(0, args.image.height - sourceH)
    const scale = Math.min(1, args.targetWidthPx / args.image.width)
    const targetW = Math.max(1, Math.round(args.image.width * scale))
    const targetH = Math.max(1, Math.round(sourceH * scale))

    const source = createCanvas(args.image.width, args.image.height)
    context2d(source).putImageData(new ImageData(args.image.data, args.image.width, args.image.height), 0, 0)
    const out = createCanvas(targetW, targetH)
    context2d(out).drawImage(source, 0, sourceY, args.image.width, sourceH, 0, 0, targetW, targetH)
    const pixels = context2d(out).getImageData(0, 0, targetW, targetH)
    return { width: pixels.width, height: pixels.height, data: pixels.data }
  }
}

function createCanvas(width: number, height: number): HTMLCanvasElement | OffscreenCanvas {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  return canvas
}

function context2d(canvas: HTMLCanvasElement | OffscreenCanvas): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d canvas unavailable')
  return ctx
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
