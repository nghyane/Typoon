/**
 * Entry-point for creating translation runs.
 *
 * Owns long-lived dependencies (vision, recognizer, translator, executor)
 * and creates lightweight TranslationRun instances per user request.
 */
import type { PageDocumentSource } from '../domain/source';
import type { TranslationRequest } from '../domain/run';
import type { VisionRuntime } from '../vision/VisionRuntime';
import type { TextRecognizer } from '../recognizers/text';
import type { Translator } from '../translators/translator';
import { TranslationRun } from './TranslationRun';
import { type PipelineConcurrency } from './StageExecutor';
import { TranslationStageSession } from './TranslationStageSession';
export declare class TranslationRuntime {
    private readonly deps;
    constructor(deps: {
        readonly vision: VisionRuntime;
        readonly recognizer: TextRecognizer;
        readonly translator: Translator;
        readonly concurrency: PipelineConcurrency;
    });
    createTranslationRun(source: PageDocumentSource, request: TranslationRequest): TranslationRun;
    createStageSession(preparation?: TranslationRequest['preparation'], runId?: `${string}-${string}-${string}-${string}-${string}`): Promise<TranslationStageSession>;
    dispose(): void;
}
