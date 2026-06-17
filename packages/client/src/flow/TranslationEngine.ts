import type { PageSource } from '../domain/source'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { TextRecognizer } from '../recognizers/text'
import type { Translator } from '../translators/translator'
import type { TranslationPostEditor } from '../translators/postEditor'
import type { SegmentRequest } from '../domain/segment'
import { SegmentTranslationRun } from './SegmentTranslationRun'
import { StageScheduler, type StageConcurrencyPolicy, type StageSchedulerOptions, type StageSchedulerPolicy } from './StageScheduler'

export const DEFAULT_STAGE_CONCURRENCY: StageConcurrencyPolicy = {
  pages: 3,
  recognize: 4,
  detect: 1,
  translate: 4,
}

export interface TranslationEngineOptions {
  readonly sourceLang?: string
  readonly targetLang: string
  readonly recognizer: TextRecognizer
  readonly detector?: TextRegionDetector
  readonly translator: Translator
  readonly postEditor?: TranslationPostEditor
  readonly scheduler?: StageSchedulerOptions
}

export interface TranslateSegmentOptions {
  readonly source: PageSource
  readonly pages?: readonly number[]
  readonly sourceLang?: string | null
  readonly targetLang?: string
  readonly postEdit?: boolean
  readonly sessionId?: string
  readonly scheduler?: StageSchedulerOptions
  readonly signal?: AbortSignal
}

export class TranslationEngine {
  constructor(private readonly options: TranslationEngineOptions) {}

  translateSegment(options: TranslateSegmentOptions): SegmentTranslationRun {
    const pageIndexes = options.pages ?? range(options.source.pageCount)

    const request: SegmentRequest = {
      pageIndexes,
      sourceLang: options.sourceLang ?? this.options.sourceLang ?? null,
      targetLang: options.targetLang ?? this.options.targetLang,
      postEdit: options.postEdit ?? false,
      sessionId: options.sessionId,
    }

    return new SegmentTranslationRun(request, {
      source: options.source,
      recognizer: this.options.recognizer,
      translator: this.options.translator,
      postEditor: this.options.postEditor,
      detector: this.options.detector,
      scheduler: new StageScheduler(resolveSchedulerPolicy(
        this.options.scheduler,
        options.scheduler,
      )),
      signal: options.signal,
    })
  }
}

function resolveSchedulerPolicy(
  base: StageSchedulerOptions | undefined,
  override: StageSchedulerOptions | undefined,
): StageSchedulerPolicy {
  return {
    concurrency: { ...DEFAULT_STAGE_CONCURRENCY, ...base?.concurrency, ...override?.concurrency },
    retry: { ...base?.retry, ...override?.retry },
  }
}

function range(length: number): number[] {
  return Array.from({ length }, (_, i) => i)
}
