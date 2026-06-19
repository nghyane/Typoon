import type { PageOverlay } from './overlay'

export type PageTranslationStatus =
  | 'queued'
  | 'loading'
  | 'preparing'
  | 'ocr'
  | 'detecting'
  | 'translating'
  | 'composing'
  | 'done'
  | 'error'

export interface TranslationProgress {
  readonly done: number
  readonly total: number
}

export interface TranslationRequest {
  readonly sourceLanguage: string | null
  readonly targetLanguage: string
  readonly scope?: 'all' | readonly number[]

  /** Source page around which to prioritise. */
  readonly priority?: { readonly aroundPageIndex: number }

  readonly preparation: PreparationStrategy
}

export type PreparationStrategy =
  | { type: 'identity' }
  | { type: 'identity-with-seams'; seamBandPx?: number }
  | { type: 'continuous-strip' }

export type TranslationRunEvent =
  | { type: 'page-status'; pageIndex: number; status: PageTranslationStatus; error?: Error }
  | { type: 'page-overlay'; overlay: PageOverlay }
  | { type: 'progress'; progress: TranslationProgress }
  | { type: 'completed'; overlays: readonly PageOverlay[] }
  | { type: 'failed'; error: Error }
  | { type: 'cancelled' }
