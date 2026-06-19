export type ReaderPhase = 'idle' | 'loading' | 'preparing' | 'ready' | 'translating' | 'done' | 'error';
export interface ReaderTranslationState {
    phase: ReaderPhase;
    prepare: {
        done: number;
        total: number;
        preparedPages: number;
    };
    translate: {
        done: number;
        total: number;
    };
    model: ReaderModelState;
    sourceLanguage: string | null;
    targetLanguage: string;
    error?: string;
}
export interface ReaderModelState {
    readonly state: 'idle' | 'resolving' | 'downloading' | 'initializing' | 'ready' | 'failed';
    readonly receivedBytes?: number;
    readonly totalBytes?: number;
    readonly ratio?: number;
    readonly error?: string;
}
type Listener = (state: ReaderTranslationState) => void;
export interface ReaderTranslationChapter {
    readonly chapterKey: string;
    readonly pageCount: number;
    readonly readPage: (index: number, signal?: AbortSignal) => Promise<Blob>;
    readonly sourceLanguage: string | null;
    readonly targetLanguage: string;
}
export declare class ReaderTranslation {
    private readonly listeners;
    private readonly recognizer;
    private translator;
    private contentHost;
    private chapter;
    private state;
    private overlay;
    private pageSizes;
    private readonly pageCache;
    private chunks;
    private readonly processedChunks;
    private readonly processingChunks;
    private abort;
    private generation;
    private overlayRevision;
    private attachedOverlayKey;
    private active;
    private renderFrame;
    private processingVisible;
    private readonly unsubscribeModelState;
    constructor();
    subscribe(fn: Listener): () => void;
    registerContentHost(host: HTMLElement): () => void;
    setChapter(chapter: ReaderTranslationChapter): void;
    clear(): void;
    translate(): void;
    cancel(): void;
    dispose(): void;
    private start;
    private loadPage;
    private ensureTranslator;
    private scheduleAttachOverlay;
    private scheduleVisibleChunkProcessing;
    private processVisibleChunks;
    private nextVisibleChunkIndex;
    private nextChunkIndex;
    private nextSequentialChunkIndex;
    private processChunk;
    private attachVisibleOverlay;
    private clearOverlay;
    private detachOverlay;
    private measureLayout;
    private refreshLayout;
    private ensurePagesForChunk;
    private bindViewportListeners;
    private unbindViewportListeners;
    private stopRun;
    private isSameChapter;
    private isCurrent;
    private setState;
    private emit;
}
export {};
