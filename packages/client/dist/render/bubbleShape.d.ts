import type { Polygon } from '../domain/geometry';
import type { FitRect } from './fitGeometry';
export type BubbleShapeKind = 'rect' | 'oval' | 'polygon' | 'tall' | 'wide';
export interface BubbleShapeProfile {
    readonly kind: BubbleShapeKind;
    readonly centerX: number;
    readonly centerY: number;
    readonly rect: FitRect;
    /** Max pixel width available for line `lineIndex` of `totalLines`. */
    widthAt(lineIndex: number, totalLines: number): number;
}
export declare function bubbleShapeProfile(polygon: Polygon, rect: FitRect): BubbleShapeProfile;
