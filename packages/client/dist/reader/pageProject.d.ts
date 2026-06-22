import type { BBox } from '../domain/geometry';
import type { TextPlacement } from '../domain/planning';
import type { PageScanUnit } from '../domain/pageScan';
import type { SafeMarginsDebug } from '../render/backgroundFit';
export type RenderTarget = 'drop' | 'page' | 'seam-below' | 'seam-above';
export interface CanvasGeometry {
    readonly captureScale: number;
    readonly haloTopPx: number;
}
/** Decide where a canvas-space placement renders, by centroid + bbox spill. */
export declare function routePlacement(canvasBBox: BBox, unit: PageScanUnit, geo: CanvasGeometry): RenderTarget;
/** Transform a canvas-space placement into page-N source space. */
export declare function canvasPlacementToSource(placement: TextPlacement, unit: PageScanUnit, geo: CanvasGeometry): TextPlacement;
export declare function canvasMarginToSource(margin: SafeMarginsDebug, geo: CanvasGeometry): SafeMarginsDebug;
/** Shift a page-source placement into seam-local space (origin at bridge top). */
export declare function shiftPlacementY(placement: TextPlacement, dy: number): TextPlacement;
export declare function shiftMarginY(margin: SafeMarginsDebug, dy: number): SafeMarginsDebug;
