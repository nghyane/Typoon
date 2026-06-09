import type { RenderedPage } from '../domain/translation'

export interface StageMetrics {
  readonly readMs?: number
  readonly recognizeMs?: number
  readonly detectMs?: number
  readonly planMs?: number
  readonly translateMs?: number
  readonly totalMs?: number
}

export type SegmentEvent =
  | { readonly type: 'segment-started'; readonly pageCount: number }
  | { readonly type: 'page-started'; readonly pageIndex: number }
  | { readonly type: 'page-ocr-ready'; readonly pageIndex: number; readonly unitCount: number; readonly metrics: StageMetrics }
  | { readonly type: 'page-text-ready'; readonly pageIndex: number; readonly page: RenderedPage; readonly metrics: StageMetrics }
  | { readonly type: 'page-layout-ready'; readonly pageIndex: number; readonly page: RenderedPage; readonly metrics: StageMetrics }
  | { readonly type: 'page-display-ready'; readonly pageIndex: number; readonly page: RenderedPage; readonly metrics: StageMetrics }
  | { readonly type: 'page-layout-failed'; readonly pageIndex: number; readonly error: unknown; readonly metrics: StageMetrics }
  | { readonly type: 'page-failed'; readonly pageIndex: number; readonly error: unknown; readonly metrics: StageMetrics }
  | { readonly type: 'segment-done'; readonly pages: readonly RenderedPage[] }
  | { readonly type: 'segment-cancelled' }

export type SegmentEventListener = (event: SegmentEvent) => void
