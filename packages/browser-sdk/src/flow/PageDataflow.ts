import type { ContextBuilder } from '../context/ContextBuilder'
import type { TextPlacement } from '../domain/planning'
import type { PageSource } from '../domain/source'
import type { RecognizedTextPage } from '../domain/text'
import type { RenderedPage } from '../domain/translation'
import type { TextRegionDetector } from '../detectors/textRegions'
import { canvasPageFromImage } from '../pipeline/canvasPageFromImage'
import { materializeRenderedPage } from '../pipeline/materializeRenderedPage'
import { layoutPlacementsFromRegions, textPlacementsFromRecognition } from '../pipeline/textPlacements'
import { textUnitsFromBlocks } from '../pipeline/textUnits'
import { translateSegment } from '../pipeline/translateSegment'
import { translationUnitsFromTextUnits } from '../pipeline/translationUnits'
import type { TextRecognizer } from '../recognizers/text'
import type { Translator } from '../translators/translator'
import type { SegmentEvent, StageMetrics } from './events'
import type { StageScheduler } from './StageScheduler'

export interface PageDataflowOptions {
  readonly workId: string
  readonly segmentId: string
  readonly pageIndex: number
  readonly source: PageSource
  readonly sourceLang: string | null
  readonly targetLang: string
  readonly recognizer: TextRecognizer
  readonly detector?: TextRegionDetector
  readonly translator: Translator
  readonly contextBuilder?: ContextBuilder
  readonly displayPolicy: DisplayPolicy
  readonly scheduler: StageScheduler
  readonly signal?: AbortSignal
  readonly emit: (event: SegmentEvent) => void
}

export type DisplayPolicy = 'layout-preferred' | 'progressive'

export class PageDataflow {
  constructor(private readonly options: PageDataflowOptions) {}

  async run(): Promise<RenderedPage> {
    const startedAt = performance.now()
    const metrics: Record<string, number> = {}
    const pageIndex = this.options.pageIndex
    this.options.emit({ type: 'page-started', pageIndex })

    const readStartedAt = performance.now()
    const input = await this.options.source.loadPage(pageIndex, this.options.signal)
    throwIfAborted(this.options.signal)
    const canvas = await canvasPageFromImage(input, { pageIndex })
    metrics.readMs = Math.round(performance.now() - readStartedAt)

    const recognizeStartedAt = performance.now()
    const recognized = await this.options.scheduler.recognize(() => this.options.recognizer.recognizeText(canvas.image, {
      sourceLang: this.options.sourceLang,
      pageIndex,
      signal: this.options.signal,
    }))
    metrics.recognizeMs = Math.round(performance.now() - recognizeStartedAt)

    const textPlanStartedAt = performance.now()
    const textUnits = textUnitsFromBlocks(recognized.blocks, pageIndex)
    const translationUnits = translationUnitsFromTextUnits(textUnits)
    const textPlacements = textPlacementsFromRecognition(recognized, textUnits)
    metrics.planMs = Math.round(performance.now() - textPlanStartedAt)
    this.options.emit({ type: 'page-ocr-ready', pageIndex, unitCount: textUnits.length, metrics: snapshotMetrics(metrics) })

    const layoutPromise = this.options.detector
      ? this.layoutPlacements(canvas.image, recognized, textUnits, startedAt, metrics)
      : Promise.resolve<readonly TextPlacement[] | null>(null)

    const context = this.options.contextBuilder
      ? await this.options.contextBuilder.buildContext({
        workId: this.options.workId,
        segmentId: this.options.segmentId,
        sourceLang: this.options.sourceLang,
        targetLang: this.options.targetLang,
        units: translationUnits,
        signal: this.options.signal,
      })
      : undefined

    const translateStartedAt = performance.now()
    const translations = await this.options.scheduler.translate(() => translateSegment({
      units: translationUnits,
      translator: this.options.translator,
      sourceLang: this.options.sourceLang,
      targetLang: this.options.targetLang,
      context,
      signal: this.options.signal,
    }))
    metrics.translateMs = Math.round(performance.now() - translateStartedAt)
    metrics.totalMs = Math.round(performance.now() - startedAt)

    const textPage = materializeRenderedPage({
      phase: 'text',
      canvas,
      recognizedText: recognized,
      textUnits,
      translationUnits,
      placements: textPlacements,
      translations,
      timingMs: metrics,
    })
    this.options.emit({ type: 'page-text-ready', pageIndex, page: textPage, metrics: snapshotMetrics(metrics) })
    if (this.options.displayPolicy === 'progressive' || !this.options.detector) {
      this.options.emit({ type: 'page-display-ready', pageIndex, page: textPage, metrics: snapshotMetrics(metrics) })
    }

    const layoutPlacements = await layoutPromise
    if (!layoutPlacements) {
      if (this.options.displayPolicy === 'layout-preferred' && this.options.detector) {
        this.options.emit({ type: 'page-display-ready', pageIndex, page: textPage, metrics: snapshotMetrics(metrics) })
      }
      return textPage
    }
    const layoutPage = materializeRenderedPage({
      phase: 'layout',
      canvas,
      recognizedText: recognized,
      textUnits,
      translationUnits,
      placements: layoutPlacements,
      translations,
      timingMs: metrics,
    })
    this.options.emit({ type: 'page-layout-ready', pageIndex, page: layoutPage, metrics: snapshotMetrics(metrics) })
    this.options.emit({ type: 'page-display-ready', pageIndex, page: layoutPage, metrics: snapshotMetrics(metrics) })
    return layoutPage
  }

  private async layoutPlacements(
    image: Awaited<ReturnType<typeof canvasPageFromImage>>['image'],
    recognized: RecognizedTextPage,
    textUnits: ReturnType<typeof textUnitsFromBlocks>,
    startedAt: number,
    metrics: Record<string, number>,
  ): Promise<readonly TextPlacement[] | null> {
    const pageIndex = this.options.pageIndex
    try {
      const detectStartedAt = performance.now()
      const regions = await this.options.scheduler.detect(() => this.options.detector!.detectTextRegions(image, { signal: this.options.signal }))
      metrics.detectMs = Math.round(performance.now() - detectStartedAt)
      const layoutPlanStartedAt = performance.now()
      const placements = layoutPlacementsFromRegions(recognized, textUnits, regions)
      metrics.planMs = Math.round((metrics.planMs ?? 0) + performance.now() - layoutPlanStartedAt)
      metrics.totalMs = Math.round(performance.now() - startedAt)
      return placements
    } catch (error) {
      this.options.emit({ type: 'page-layout-failed', pageIndex, error, metrics: snapshotMetrics(metrics) })
      return null
    }
  }
}

function snapshotMetrics(metrics: Record<string, number>): StageMetrics {
  return { ...metrics }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
