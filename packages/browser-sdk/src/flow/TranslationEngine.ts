import type { ContextBuilder } from '../context/ContextBuilder'
import type { PageSource } from '../domain/source'
import type { TextRegionDetector } from '../detectors/textRegions'
import type { TextRecognizer } from '../recognizers/text'
import type { Translator } from '../translators/translator'
import type { DisplayPolicy } from './PageDataflow'
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
  readonly contextBuilder?: ContextBuilder
  readonly displayPolicy?: DisplayPolicy
  readonly scheduler?: StageSchedulerOptions
}

export interface TranslateSegmentOptions {
  readonly workId: string
  readonly segmentId: string
  readonly source: PageSource
  readonly sourceLang?: string | null
  readonly targetLang?: string
  readonly pages?: readonly number[]
  readonly scheduler?: StageSchedulerOptions
  readonly displayPolicy?: DisplayPolicy
  readonly stopOnError?: boolean
  readonly signal?: AbortSignal
}

export class TranslationEngine {
  constructor(private readonly options: TranslationEngineOptions) {}

  translateSegment(options: TranslateSegmentOptions): SegmentTranslationRun {
    return new SegmentTranslationRun({
      workId: options.workId,
      segmentId: options.segmentId,
      source: options.source,
      sourceLang: options.sourceLang ?? this.options.sourceLang ?? null,
      targetLang: options.targetLang ?? this.options.targetLang,
      recognizer: this.options.recognizer,
      detector: this.options.detector,
      translator: this.options.translator,
      contextBuilder: this.options.contextBuilder,
      displayPolicy: options.displayPolicy ?? this.options.displayPolicy ?? 'layout-preferred',
      scheduler: new StageScheduler(resolveSchedulerPolicy(this.options.scheduler, options.scheduler)),
      pages: options.pages,
      stopOnError: options.stopOnError,
      signal: options.signal,
    })
  }
}

function resolveSchedulerPolicy(base: StageSchedulerOptions | undefined, override: StageSchedulerOptions | undefined): StageSchedulerPolicy {
  return {
    concurrency: { ...DEFAULT_STAGE_CONCURRENCY, ...base?.concurrency, ...override?.concurrency },
    retry: { ...base?.retry, ...override?.retry },
  }
}
