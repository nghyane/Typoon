import type { ImageInput } from '../image/input'

export interface PageSource {
  readonly pageCount: number
  loadPage(index: number, signal?: AbortSignal): ImageInput | Promise<ImageInput>
}
