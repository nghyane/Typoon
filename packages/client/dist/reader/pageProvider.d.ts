import type { SourcePageSize } from '../pipeline/chapterContent';
export interface LoadedPage {
    readonly index: number;
    readonly blob: Blob;
    readonly size: SourcePageSize;
}
export type ReadPageFn = (index: number, signal?: AbortSignal) => Promise<Blob>;
export interface PageProviderOptions {
    readonly pageCount: number;
    readonly maxCachedPages: number;
    readonly readPage: ReadPageFn;
    readonly onProgress?: (loadedPages: number) => void;
}
export declare class PageProvider {
    private readonly options;
    private readonly cache;
    private readonly order;
    private readonly sizes;
    private loadedCount;
    constructor(options: PageProviderOptions);
    size(index: number): SourcePageSize | null;
    read(index: number, signal: AbortSignal): Promise<LoadedPage>;
    /** Evict LRU pages beyond the cap, keeping the given indexes resident. */
    evictExcept(keep: Iterable<number>): void;
    clear(): void;
    private touch;
}
