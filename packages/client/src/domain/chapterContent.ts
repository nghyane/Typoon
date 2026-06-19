import type { TextPlacement } from './planning'
import type { TranslatedUnit } from './translation'

export interface ChapterContentRect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

export interface ChapterContentPage {
  readonly pageIndex: number
  readonly sourceSize: { readonly width: number; readonly height: number }
  readonly contentRect: ChapterContentRect
}

export interface ChapterContentLayout {
  readonly contentSize: { readonly width: number; readonly height: number }
  readonly pages: readonly ChapterContentPage[]
}

export interface ChapterOverlay {
  readonly contentSize: { readonly width: number; readonly height: number }
  readonly placements: readonly TextPlacement[]
  readonly translations: readonly TranslatedUnit[]
}
