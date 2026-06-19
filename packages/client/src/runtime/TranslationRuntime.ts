/**
 * Entry-point for creating translation runs.
 *
 * Owns long-lived dependencies (vision, recognizer, translator, executor)
 * and creates lightweight TranslationRun instances per user request.
 */

import type { PageDocumentSource } from '../domain/source'
import type { TranslationRequest } from '../domain/run'
import type { VisionRuntime } from '../vision/VisionRuntime'
import type { TextRecognizer } from '../recognizers/text'
import type { Translator } from '../translators/translator'
import { TranslationRun } from './TranslationRun'
import { StageExecutor, type PipelineConcurrency } from './StageExecutor'
import { TranslationStageSession } from './TranslationStageSession'

export class TranslationRuntime {
  private readonly deps: {
    readonly vision: VisionRuntime
    readonly recognizer: TextRecognizer
    readonly translator: Translator
    readonly concurrency: PipelineConcurrency
  }

  constructor(
    deps: {
      readonly vision: VisionRuntime
      readonly recognizer: TextRecognizer
      readonly translator: Translator
      readonly concurrency: PipelineConcurrency
    },
  ) {
    this.deps = deps
  }

  createTranslationRun(
    source: PageDocumentSource,
    request: TranslationRequest,
  ): TranslationRun {
    return new TranslationRun(source, request, {
      vision: this.deps.vision,
      recognizer: this.deps.recognizer,
      translator: this.deps.translator,
      executor: new StageExecutor(this.deps.concurrency),
      concurrency: this.deps.concurrency,
    })
  }

  async createStageSession(
    preparation: TranslationRequest['preparation'] = { type: 'identity' },
    runId = crypto.randomUUID(),
  ): Promise<TranslationStageSession> {
    const session = await this.deps.vision.beginPreparation(runId, preparation)
    return new TranslationStageSession(runId, session, {
      vision: this.deps.vision,
      recognizer: this.deps.recognizer,
      translator: this.deps.translator,
    })
  }

  dispose(): void {
    this.deps.vision.dispose()
  }
}
