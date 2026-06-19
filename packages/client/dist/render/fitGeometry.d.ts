import type { BBox } from '../domain/geometry';
import type { TextPlacement } from '../domain/planning';
export interface FitRect {
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
    readonly rotationDeg: number;
}
export declare function drawableRect(placement: TextPlacement): FitRect;
/** Fit rect for single-block placements: center on OCR text boxes, sized to grow room. */
export declare function textFitRect(placement: TextPlacement): FitRect;
export declare function rectBBox(rect: FitRect): BBox;
export declare function rectFromBBox(bbox: BBox, rotationDeg?: number): FitRect;
