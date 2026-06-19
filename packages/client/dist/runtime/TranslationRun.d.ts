/**
 * Page-local translation runtime.
 *
 * Source pages are the progress source of truth:
 *   load -> identity prepare -> OCR -> translate -> compose -> overlay/done
 *
 * Cross-page OCR is a repair pass only.  A seam repair image is created after
 * both adjacent source pages are prepared, then OCR/translated best-effort and
 * projected back onto the two source pages.  It never blocks source page
 * progress, but the run waits for bounded seam tasks before completing so their
 * overlays can still attach.
 */
import type { TranslationRequest, TranslationRunEvent } from '../domain/run';
import type { PageOverlay } from '../domain/overlay';
import type { PageDocumentSource } from '../domain/source';
import type { VisionRuntime } from '../vision/VisionRuntime';
import type { TextRecognizer } from '../recognizers/text';
import type { Translator } from '../translators/translator';
import { StageExecutor, type PipelineConcurrency } from './StageExecutor';
export declare class TranslationRun {
    private readonly runId;
    private readonly abort;
    private readonly listeners;
    private readonly pageStatuses;
    private readonly overlays;
    private lastProgress;
    private readonly source;
    private readonly request;
    private readonly deps;
    private started;
    private finished;
    private doneResolve;
    private doneReject;
    readonly done: Promise<readonly PageOverlay[]>;
    constructor(source: PageDocumentSource, request: TranslationRequest, deps: {
        readonly vision: VisionRuntime;
        readonly recognizer: TextRecognizer;
        readonly translator: Translator;
        readonly executor: StageExecutor;
        readonly concurrency: PipelineConcurrency;
    });
    subscribe(listener: (event: TranslationRunEvent) => void): () => void;
    start(): void;
    cancel(reason?: Error): void;
    private run;
    private runContinuousStrip;
    private processContinuousPreparedPage;
    private processSourcePage;
    private processSeamRepair;
    private processPreparedPage;
    private emitProjectionStatuses;
    private emitStatus;
    private emitMergedOverlay;
    private markPageError;
    private setInitialPageStatuses;
    private emitProgress;
    private emit;
    private finishCompleted;
    private finishCancelled;
    private finishFailed;
}
