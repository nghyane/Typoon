import type { CanvasPage } from '../domain/canvas'
import type { PagePlan } from '../domain/planning'
import type { TranslatedPage, TranslatedUnit } from '../domain/translation'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { ImageInput } from '../image/input'
import type { TextRecognizer } from '../recognizers/text'
import { canvasPageFromImage } from './canvasPageFromImage'
import { planPage } from './planPage'

export interface MaterializedPagePlan {
  readonly canvas: CanvasPage
  readonly recognizedText: Awaited<ReturnType<TextRecognizer['recognizeText']>>
  readonly plan: PagePlan
}

export async function materializePagePlan(input: ImageInput, options: {
  readonly recognizer: TextRecognizer
  readonly detector?: TextRegionDetector
  readonly sourceLang: string | null
  readonly pageIndex: number
  readonly signal?: AbortSignal
}): Promise<MaterializedPagePlan> {
  const canvas = await canvasPageFromImage(input, { pageIndex: options.pageIndex })
  throwIfAborted(options.signal)
  const recognizedPromise = options.recognizer.recognizeText(canvas.image, {
    sourceLang: options.sourceLang,
    pageIndex: options.pageIndex,
    signal: options.signal,
  })
  const regionsPromise = options.detector
    ? options.detector.detectTextRegions(canvas.image, { signal: options.signal })
    : Promise.resolve([])
  const [recognizedText, regions] = await Promise.all([recognizedPromise, regionsPromise])
  throwIfAborted(options.signal)
  const plan = planPage({ text: recognizedText, regions, pageIndex: options.pageIndex })
  return { canvas, recognizedText, plan }
}

export function translatedPageFromMaterializedPlan(
  artifact: MaterializedPagePlan,
  translations: readonly TranslatedUnit[],
): TranslatedPage {
  return {
    image: artifact.canvas.image,
    pageIndex: artifact.canvas.pageIndex,
    pageSize: artifact.plan.pageSize,
    detectedLanguage: artifact.recognizedText.detectedLanguage,
    placements: artifact.plan.placements,
    units: artifact.plan.units,
    translations,
    timingMs: { ...artifact.recognizedText.timingMs, ...artifact.plan.timingMs },
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}
