import type { ReaderPageOverlay } from '../domain/pageScan';
export interface OverlayChapterMeta {
    readonly sourceLanguage: string | null;
    readonly targetLanguage: string | null;
}
export declare class OverlayManager {
    private readonly marginPx;
    private host;
    private overlays;
    private revision;
    private meta;
    private hidden;
    private readonly pageEls;
    private readonly attached;
    private readonly seams;
    private readonly visible;
    private observer;
    private hostObserver;
    constructor(marginPx: number);
    setHost(host: HTMLElement | null): void;
    get currentHost(): HTMLElement | null;
    /** Register a page element (called from a Svelte action). Returns cleanup. */
    registerPage(pageIndex: number, el: HTMLElement): () => void;
    /** Update overlay data; re-attaches only visible pages at the new revision. */
    update(overlays: Map<number, ReaderPageOverlay>, revision: number, meta: OverlayChapterMeta): void;
    setHidden(hidden: boolean): void;
    detach(): void;
    dispose(): void;
    private ensureObserver;
    private attachVisible;
    private attachPageSurface;
    private attachSeam;
    private repositionSeams;
    /**
     * Position a seam bridge from its owning page's live rect.
     * seamSize is in the owner page's source px; the owner occupies
     *   [topOffsetSource, topOffsetSource + ownerSourceHeight) of the seam band.
     * displayScale is derived from the owner's display width (all pages share it).
     */
    private positionBridge;
    private removeSeamsOwnedBy;
}
