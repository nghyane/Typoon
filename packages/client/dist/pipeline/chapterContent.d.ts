import type { ChapterContentLayout, ChapterContentPage, ChapterContentRect } from '../domain/chapterContent';
import type { RecognizedTextPage } from '../domain/text';
export interface SourcePageSize {
    readonly width: number;
    readonly height: number;
}
export interface ChapterOcrChunk {
    readonly index: number;
    readonly contentRect: ChapterContentRect;
    readonly coreRect: ChapterContentRect;
}
export declare function buildChapterContentLayout(pageSizes: readonly SourcePageSize[]): ChapterContentLayout;
export declare function chapterOcrChunks(layout: ChapterContentLayout): readonly ChapterOcrChunk[];
export declare function pagesIntersectingChunk(layout: ChapterContentLayout, chunk: ChapterOcrChunk): readonly ChapterContentPage[];
export declare function mapChunkRecognitionToChapter(recognized: RecognizedTextPage, chunk: ChapterOcrChunk): RecognizedTextPage;
export declare function mergeChunkRecognitions(chunks: readonly RecognizedTextPage[], layout: ChapterContentLayout): RecognizedTextPage;
export declare function rectIntersection(a: ChapterContentRect, b: ChapterContentRect): ChapterContentRect | null;
