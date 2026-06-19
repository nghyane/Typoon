import type { PageSource } from '../domain/source';
import type { TextRegionDetector } from '../detectors/textRegions';
import type { TextRecognizer } from '../recognizers/text';
import type { Translator } from '../translators/translator';
import type { TranslationPostEditor } from '../translators/postEditor';
import { SegmentTranslationRun } from './SegmentTranslationRun';
import { type StageConcurrencyPolicy, type StageSchedulerOptions } from './StageScheduler';
export declare const DEFAULT_STAGE_CONCURRENCY: StageConcurrencyPolicy;
export interface SegmentDisplayOptions {
    readonly progressive?: boolean;
}
export interface TranslationEngineOptions {
    readonly sourceLang?: string;
    readonly targetLang: string;
    readonly recognizer: TextRecognizer;
    readonly detector?: TextRegionDetector;
    readonly translator: Translator;
    readonly postEditor?: TranslationPostEditor;
    readonly scheduler?: StageSchedulerOptions;
    readonly display?: SegmentDisplayOptions;
}
export interface TranslateSegmentOptions {
    readonly source: PageSource;
    readonly pages?: readonly number[];
    readonly sourceLang?: string | null;
    readonly targetLang?: string;
    readonly postEdit?: boolean;
    readonly sessionId?: string;
    readonly scheduler?: StageSchedulerOptions;
    readonly display?: SegmentDisplayOptions;
    readonly signal?: AbortSignal;
}
export declare class TranslationEngine {
    private readonly options;
    constructor(options: TranslationEngineOptions);
    translateSegment(options: TranslateSegmentOptions): SegmentTranslationRun;
}
