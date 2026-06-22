import type { PageScanUnit, ReaderPageOverlay } from '../domain/pageScan';
import { LensTextRecognizer } from '../recognizers/lens/LensTextRecognizer';
import type { Translator } from '../translators/translator';
import type { LoadedPage } from './pageProvider';
import type { TranslationConfig } from './translationConfig';
export interface PagePipelineDeps {
    readonly recognizer: LensTextRecognizer;
    readonly translator: () => Translator;
    readonly config: TranslationConfig;
}
export interface PagePipelineArgs {
    readonly unit: PageScanUnit;
    readonly loadPage: (index: number) => Promise<LoadedPage>;
    readonly sourceLanguage: string | null;
    readonly targetLanguage: string;
    readonly signal: AbortSignal;
}
export declare class PagePipeline {
    private readonly deps;
    constructor(deps: PagePipelineDeps);
    run(args: PagePipelineArgs): Promise<ReaderPageOverlay>;
    private route;
}
