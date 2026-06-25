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
    /** Capture core+halo, OCR, detect regions, drop noise, derive role context. */
    private scanPage;
    /** Adapt the recognizer to the per-bubble crop interface Phase B needs. */
    private cropRecognizer;
    /** Build overlay placements + synthetic translation units; null if nothing to do. */
    private composePlacements;
    /** Translate the composed units and return the translated payloads. */
    private translatePlacements;
    private route;
}
export declare function deduplicateSeamBlocks(overlays: Map<number, ReaderPageOverlay>): void;
