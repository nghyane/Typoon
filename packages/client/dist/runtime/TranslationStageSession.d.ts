import type { PageOverlay } from '../domain/overlay';
import type { PreparedPageHandle } from '../domain/prepared';
import type { TranslationRequest } from '../domain/run';
import type { PageAsset } from '../domain/source';
import type { TextRegion } from '../domain/regions';
import type { VisionRuntime, PreparationSession } from '../vision/VisionRuntime';
import type { TextRecognizer } from '../recognizers/text';
import type { Translator } from '../translators/translator';
import { type PreparedTextResult, type PreparedTranslationResult } from '../pipeline/translatePreparedPage';
export interface TranslationStageSessionDeps {
    readonly vision: VisionRuntime;
    readonly recognizer: TextRecognizer;
    readonly translator: Translator;
}
export declare class TranslationStageSession {
    readonly runId: string;
    private readonly deps;
    private readonly session;
    private disposed;
    constructor(runId: string, session: PreparationSession, deps: TranslationStageSessionDeps);
    preparePage(asset: PageAsset, signal?: AbortSignal): Promise<readonly PreparedPageHandle[]>;
    flush(signal?: AbortSignal): Promise<readonly PreparedPageHandle[]>;
    createSeamRepair(top: PreparedPageHandle, bottom: PreparedPageHandle, bandPx?: number, signal?: AbortSignal): Promise<PreparedPageHandle | null>;
    recognize(handle: PreparedPageHandle, sourceLanguage: string | null, signal?: AbortSignal): Promise<PreparedTextResult>;
    translate(text: PreparedTextResult, sourceLanguage: string | null, targetLanguage: string, signal?: AbortSignal): Promise<PreparedTranslationResult>;
    composeOverlay(handle: PreparedPageHandle, translated: PreparedTranslationResult, regions?: readonly TextRegion[] | null, signal?: AbortSignal): Promise<readonly PageOverlay[]>;
    recognizeTranslateCompose(args: {
        readonly handle: PreparedPageHandle;
        readonly sourceLanguage: string | null;
        readonly targetLanguage: string;
        readonly regions?: readonly TextRegion[] | null;
        readonly signal?: AbortSignal;
    }): Promise<readonly PageOverlay[]>;
    release(handle: PreparedPageHandle): void;
    dispose(): void;
    private assertActive;
}
export type RuntimeStageRequest = Pick<TranslationRequest, 'preparation'>;
