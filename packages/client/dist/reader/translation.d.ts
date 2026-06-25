import type { SourcePageSize } from '../pipeline/chapterContent';
import { type TranslationConfig } from './translationConfig';
import { type ReaderModelState } from './visionRuntime';
export type ReaderPhase = 'idle' | 'loading' | 'ready' | 'translating' | 'done' | 'error';
/** Translation backend the reader uses. AI/custom LLM gateway is future work. */
export type TranslationProvider = 'deepl' | 'google';
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
    hidden: boolean;
    /** Chunks skipped after exhausting retries (partial translation). */
    failed: number;
    error?: string;
}
export type { ReaderModelState };
type Listener = (state: ReaderTranslationState) => void;
export interface ReaderTranslationChapter {
    readonly chapterKey: string;
    readonly pageCount: number;
    readonly readPage: (index: number, signal?: AbortSignal) => Promise<Blob>;
    readonly sourceLanguage: string | null;
    readonly targetLanguage: string;
    /**
     * Authoritative source size per page, when the host already decoded it (e.g.
     * the reader's page cache). Lets the provider skip a redundant decode and, more
     * importantly, makes `unit.source` identical to the size the host renders the
     * page frame with — so overlay geometry and the displayed image never diverge.
     */
    readonly pageSize?: (index: number) => SourcePageSize | null;
}
export interface ReaderTranslationOptions {
    readonly config?: TranslationConfig;
    readonly provider?: TranslationProvider;
}
export declare class ReaderTranslation {
    private readonly listeners;
    private readonly recognizer;
    private readonly overlays;
    private readonly scheduler;
    private readonly config;
    private readonly pipeline;
    private translator;
    private translatorProvider;
    private provider;
    private chapter;
    private pages;
    private state;
    private pageOverlays;
    private units;
    private abort;
    private generation;
    private overlayRevision;
    private active;
    private draining;
    private latestModel;
    private readonly unsubscribeModelState;
    constructor(options?: ReaderTranslationOptions);
    subscribe(fn: Listener): () => void;
    registerContentHost(host: HTMLElement): () => void;
    /** Register a page element (called from a Svelte action). Returns cleanup. */
    registerPage(pageIndex: number, el: HTMLElement): () => void;
    setChapter(chapter: ReaderTranslationChapter): void;
    clear(): void;
    translate(): void;
    /** Toggle translation visibility. Fixes the broken hide behavior. */
    setHidden(hidden: boolean): void;
    /** Choose the translation backend. Applies to the next translate() run. */
    setProvider(provider: TranslationProvider): void;
    cancel(): void;
    dispose(): void;
    private start;
    private drain;
    private processPage;
    private rebuildPlan;
    private resolveUnit;
    private measure;
    private measuredPages;
    private pageSizeFor;
    private visibleRange;
    private evictPages;
    private ensureTranslator;
    private disposeTranslator;
    private syncOverlay;
    private clearOverlay;
    private clearOverlayKeepActive;
    private stopRun;
    private isSameChapter;
    private isCurrent;
    private setState;
    private emit;
}
