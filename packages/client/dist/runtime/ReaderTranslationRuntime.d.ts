import type { PageDocumentSource } from '../domain/source';
import type { PreparedChapter } from '../domain/preparedChapter';
import type { PrepareArtifactSink, PrepareProfile, PrepareStrategy } from '../domain/prepare';
import type { TranslationRequest } from '../domain/run';
import type { VisionRuntime } from '../vision/VisionRuntime';
import type { TextRecognizer } from '../recognizers/text';
import type { Translator } from '../translators/translator';
import type { PipelineConcurrency } from './StageExecutor';
export interface ReaderChapterPrepareOptions {
    readonly runId?: string;
    readonly strategy?: PrepareStrategy;
    readonly profile?: PrepareProfile;
    readonly artifacts?: PrepareArtifactSink;
    readonly signal?: AbortSignal;
}
export type ReaderTranslationRequest = Omit<TranslationRequest, 'preparation'>;
export declare function prepareReaderChapter(source: PageDocumentSource, options?: ReaderChapterPrepareOptions): Promise<PreparedChapter>;
export declare class ReaderTranslationRuntime {
    private readonly runtime;
    constructor(deps: {
        readonly vision: VisionRuntime;
        readonly recognizer: TextRecognizer;
        readonly translator: Translator;
        readonly concurrency: PipelineConcurrency;
    });
    createTranslationRun(chapter: PreparedChapter, request: ReaderTranslationRequest): import("./TranslationRun").TranslationRun;
    dispose(): void;
}
