import { BrowserSdkError } from '../errors'
import type { ImagePixels } from '../domain/image'

export type ImageInput = HTMLImageElement | HTMLCanvasElement | ImageBitmap | Blob | File | string

export async function readImageInput(input: ImageInput): Promise<ImagePixels> {
  if (typeof input === 'string') {
    const res = await fetch(input)
    if (!res.ok) throw new BrowserSdkError('IMAGE_DECODE_FAILED', `image fetch failed: ${res.status}`)
    return readImageInput(await res.blob())
  }

  if (input instanceof Blob) {
    try {
      return await readBitmap(await createImageBitmap(input))
    } catch (cause) {
      throw new BrowserSdkError('IMAGE_DECODE_FAILED', 'failed to decode image blob', cause)
    }
  }

  if (input instanceof HTMLImageElement) {
    if (!input.complete) await input.decode()
    return readDrawable(input, input.naturalWidth, input.naturalHeight)
  }

  if (input instanceof HTMLCanvasElement) {
    return readCanvas(input)
  }

  return readBitmap(input)
}

async function readBitmap(bitmap: ImageBitmap): Promise<ImagePixels> {
  const canvas = document.createElement('canvas')
  canvas.width = bitmap.width
  canvas.height = bitmap.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new BrowserSdkError('IMAGE_DECODE_FAILED', '2d canvas unavailable')
  ctx.drawImage(bitmap, 0, 0)
  bitmap.close()
  return readCanvas(canvas)
}

function readDrawable(
  source: CanvasImageSource,
  width: number,
  height: number,
): ImagePixels {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new BrowserSdkError('IMAGE_DECODE_FAILED', '2d canvas unavailable')
  ctx.drawImage(source, 0, 0, width, height)
  return readCanvas(canvas)
}

function readCanvas(canvas: HTMLCanvasElement): ImagePixels {
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new BrowserSdkError('IMAGE_DECODE_FAILED', '2d canvas unavailable')
  try {
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height)
    return { width: image.width, height: image.height, data: image.data }
  } catch (cause) {
    throw new BrowserSdkError('IMAGE_CORS_TAINTED', 'image pixels are not readable; check CORS/proxy', cause)
  }
}
