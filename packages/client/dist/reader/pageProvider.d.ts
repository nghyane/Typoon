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
    /**
     * Optional authoritative source size per page. When provided, it is the single
     * source of truth for page geometry: the provider skips its own decode so that
     * `unit.source` (the overlay's % denominator) is byte-identical to the size
     * the renderer uses for the page frame's aspect ratio. Falls back to decoding
     * when it returns null.
     */
    readonly pageSize?: (index: number) => SourcePageSize | null;
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
    /** Preload image dimensions for all pages without keeping blobs in cache. */
    preloadSizes(signal: AbortSignal): Promise<void>;
    clear(): void;
    private touch;
}
