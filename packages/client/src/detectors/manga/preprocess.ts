import type * as ort from 'onnxruntime-web/wasm'
import type { ImagePixels } from '../../domain/image'
import type { OrtModule } from '../../models/OrtBackend'
import { COMIC_DETR_INPUT_SIZE } from './ortTypes'

export function createFeeds(ortModule: OrtModule, image: ImagePixels): Record<string, ort.Tensor> {
  return {
    images: new ortModule.Tensor('float32', prepareImageTensor(image), [1, 3, COMIC_DETR_INPUT_SIZE, COMIC_DETR_INPUT_SIZE]),
    orig_target_sizes: new ortModule.Tensor('int64', new BigInt64Array([BigInt(image.width), BigInt(image.height)]), [1, 2]),
  }
}

/** Canvas resize + CHW normalization, independent of the ORT module so it can
 *  run on the main thread and ship the planar tensor to a worker. */
export function prepareImageTensor(image: ImagePixels): Float32Array {
  const source = document.createElement('canvas')
  source.width = image.width
  source.height = image.height
  const sourceCtx = source.getContext('2d')
  if (!sourceCtx) throw new Error('2d canvas unavailable')
  sourceCtx.putImageData(new ImageData(image.data, image.width, image.height), 0, 0)

  const canvas = document.createElement('canvas')
  canvas.width = COMIC_DETR_INPUT_SIZE
  canvas.height = COMIC_DETR_INPUT_SIZE
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2d canvas unavailable')
  ctx.drawImage(source, 0, 0, COMIC_DETR_INPUT_SIZE, COMIC_DETR_INPUT_SIZE)
  const pixels = ctx.getImageData(0, 0, COMIC_DETR_INPUT_SIZE, COMIC_DETR_INPUT_SIZE).data
  const out = new Float32Array(1 * 3 * COMIC_DETR_INPUT_SIZE * COMIC_DETR_INPUT_SIZE)
  const plane = COMIC_DETR_INPUT_SIZE * COMIC_DETR_INPUT_SIZE
  for (let i = 0; i < plane; i++) {
    out[i] = (pixels[i * 4] ?? 0) / 255
    out[plane + i] = (pixels[i * 4 + 1] ?? 0) / 255
    out[plane * 2 + i] = (pixels[i * 4 + 2] ?? 0) / 255
  }
  return out
}
