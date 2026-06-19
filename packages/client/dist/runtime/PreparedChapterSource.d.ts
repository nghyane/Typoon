import type { PreparedChapter } from '../domain/preparedChapter';
import type { PageDocumentSource } from '../domain/source';
export declare class PreparedChapterSource implements PageDocumentSource {
    readonly pageCount: number;
    private readonly chapter;
    constructor(chapter: PreparedChapter);
    readPage(index: number, signal?: AbortSignal): Promise<{
        index: number;
        pixels: import("..").ImagePixels;
        size: import("..").PageSize;
        projections: readonly import("../domain/preparedChapter").PageProjection[];
    }>;
}
