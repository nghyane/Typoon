import type { BBox } from './geometry';
import type { TextPlacement } from './planning';
import type { PageSize } from './source';
import type { TranslatedUnit } from './translation';
import type { SafeMarginsDebug } from '../render/backgroundFit';
/** A stateless scan unit. identity === pageIndex. */
export interface PageScanUnit {
    readonly pageIndex: number;
    readonly source: PageSize;
    readonly prevIndex: number | null;
    readonly nextIndex: number | null;
    readonly haloTopPx: number;
    readonly haloBottomPx: number;
}
export interface PlacementItem {
    readonly placement: TextPlacement;
    readonly margin: SafeMarginsDebug;
}
/**
 * A bubble that crosses the seam between two stacked pages. Rendered on a bridge
 * overlay that spans both page elements (not clipped to either). Coordinates are
 * seam-local: origin at the top of the top page, y increases downward through
 * the top page and into the bottom page's strip — i.e. the same space as the
 * owning page's capture canvas (minus halo offset), so no extra transform.
 */
export interface SeamOverlay {
    readonly topPageIndex: number;
    readonly bottomPageIndex: number;
    /** seam-local canvas size: [width, topPageSourceHeight + neighborHaloPx]. */
    readonly seamSize: readonly [number, number];
    /** y (seam-local px) where the bottom page begins (= top page source height). */
    readonly seamSplitY: number;
    readonly items: readonly PlacementItem[];
    readonly translations: readonly TranslatedUnit[];
}
/** Per-page overlay: page-local items + optional seam bridges above/below. */
export interface ReaderPageOverlay {
    readonly pageIndex: number;
    readonly pageSize: readonly [number, number];
    readonly items: readonly PlacementItem[];
    readonly translations: readonly TranslatedUnit[];
    readonly seamBelow: SeamOverlay | null;
    readonly seamAbove: SeamOverlay | null;
}
export declare function emptyReaderPageOverlay(pageIndex: number, source: PageSize): ReaderPageOverlay;
export declare function bboxCentroidY(bbox: BBox): number;
export declare function bboxCentroidX(bbox: BBox): number;
