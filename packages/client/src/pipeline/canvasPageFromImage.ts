import type { CanvasPage } from '../domain/canvas'
import type { ImageInput } from '../image/input'
import { readImageInput } from '../image/input'

export async function canvasPageFromImage(input: ImageInput, options: {
  readonly pageIndex: number
}): Promise<CanvasPage> {
  const image = await readImageInput(input)
  return {
    id: `p${options.pageIndex}`,
    pageIndex: options.pageIndex,
    pageSize: [image.width, image.height],
    image,
    sourcePageIndices: [options.pageIndex],
  }
}
