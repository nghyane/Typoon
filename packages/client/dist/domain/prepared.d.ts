/**
 * Prepared page types.
 *
 * PreparedPageHandle is the main-thread proxy for image data held in the
 * vision worker.  The handle carries enough metadata for the translation
 * pipeline to work, but the pixel data stays on the worker side and is only
 * fetched on demand via `VisionRuntime.readPixels()`.
 */
import type { PageSize, Rect } from './source';
export interface PageProjection {
    /** Source/display page that contributed to this prepared page. */
    readonly sourcePageIndex: number;
    /** Region within the prepared image. */
    readonly preparedRect: Rect;
    /** Corresponding region in the source image (often identical for identity). */
    readonly sourceRect: Rect;
    /** Full pixel dimensions of the source/display page. */
    readonly sourcePageSize?: PageSize;
}
export interface PreparedPageHandle {
    /** Source page or auxiliary repair image. Defaults to source-page. */
    readonly kind?: 'source-page' | 'seam-repair';
    /** Globally-unique id for this translation run. */
    readonly runId: string;
    /** Unique within this run. */
    readonly preparedPageId: string;
    /** Index of this prepared page in the prepared sequence. */
    readonly preparedPageIndex: number;
    /** Pixel dimensions of the prepared image. */
    readonly size: PageSize;
    /** How this prepared page maps back to source pages. */
    readonly projections: readonly PageProjection[];
}
