import type { BBox } from '../domain/geometry';
import type { ImagePixels } from '../domain/image';
import type { TextPlacement } from '../domain/planning';
import type { FitRect } from './fitGeometry';
export type Rgb = readonly [number, number, number];
export interface SafeMargins {
    readonly top: number;
    readonly bottom: number;
    readonly left: number;
    readonly right: number;
}
export interface SafeMarginsDebug {
    readonly reasons: {
        readonly top: string;
        readonly bottom: string;
        readonly left: string;
        readonly right: string;
        readonly overall: string;
    };
    readonly margins: SafeMargins;
    readonly backgroundRgb: Rgb | null;
    readonly backgroundTolerance: number;
    readonly safeBounds: BBox;
    readonly componentBBox: BBox | null;
    readonly componentConfidence: number;
    readonly shape: SafeShapeProfile | null;
}
export interface SafeShapeSpan {
    readonly y: number;
    readonly x1: number;
    readonly x2: number;
}
export interface SafeShapeProfile {
    readonly bounds: BBox;
    readonly spans: readonly SafeShapeSpan[];
    readonly confidence: number;
}
export declare function hasReliableBackgroundFill(margin: SafeMarginsDebug | null | undefined): margin is SafeMarginsDebug & {
    readonly backgroundRgb: Rgb;
};
export declare function hasAnyBackgroundFill(margin: SafeMarginsDebug | null | undefined): margin is SafeMarginsDebug & {
    readonly backgroundRgb: Rgb;
};
export declare function estimateSafeMargins(args: {
    readonly image: ImagePixels;
    readonly placement: TextPlacement;
    readonly baseRect: FitRect;
    readonly obstacles: readonly BBox[];
    readonly pageSize: readonly [number, number];
}): SafeMarginsDebug;
