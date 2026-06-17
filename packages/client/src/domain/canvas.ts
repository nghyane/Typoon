import type { ImagePixels } from './image'

export interface CanvasPage {
  readonly id: string
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly image: ImagePixels
  readonly sourcePageIndices: readonly number[]
}

export interface CanvasWindow {
  readonly id: string
  readonly pageIndex: number
  readonly pageSize: readonly [number, number]
  readonly image: ImagePixels
  readonly sourcePageIndices: readonly number[]
}
