import type { ChapterContentLayout, ChapterOverlay } from '../domain/chapterContent';
import type { ImagePixels } from '../domain/image';
import type { TextPlacement } from '../domain/planning';
import type { TextRegion } from '../domain/regions';
import type { RecognizedTextPage } from '../domain/text';
import type { TranslatedUnit } from '../domain/translation';
import type { Translator } from '../translators/translator';
import type { SafeMarginsDebug } from '../render/backgroundFit';
import type { ChapterOcrChunk } from './chapterContent';
export interface ChapterContentOverlay extends ChapterOverlay {
    readonly placementMargins: readonly SafeMarginsDebug[];
}
export interface ChapterOverlaySlice {
    readonly placements: readonly TextPlacement[];
    readonly translations: readonly TranslatedUnit[];
    readonly placementMargins: readonly SafeMarginsDebug[];
}
export interface OverlayPlacementItem {
    readonly placement: TextPlacement;
    readonly margin: SafeMarginsDebug;
}
export interface TranslateChapterContentChunkArgs {
    readonly recognized: RecognizedTextPage;
    readonly chunk: ChapterOcrChunk;
    readonly layout: ChapterContentLayout;
    readonly image: ImagePixels;
    readonly regions?: readonly TextRegion[] | null;
    readonly translator: () => Translator;
    readonly sourceLanguage: string | null;
    readonly targetLanguage: string;
    readonly signal?: AbortSignal;
}
export declare function emptyChapterContentOverlay(layout: ChapterContentLayout): ChapterContentOverlay;
export declare function translateChapterContentChunk(args: TranslateChapterContentChunkArgs): Promise<ChapterOverlaySlice>;
export declare function mergeChapterContentOverlay(existing: ChapterContentOverlay, incoming: ChapterOverlaySlice): ChapterContentOverlay;
