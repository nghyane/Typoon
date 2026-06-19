import type { TextPlacement } from '../domain/planning';
import { type SafeMarginsDebug } from './backgroundFit';
export type EraseStrategy = 'none' | 'flat-fill';
export type ErasePlan = {
    readonly kind: 'none';
} | {
    readonly kind: 'flat-fill';
    readonly shapes: readonly EraseShape[];
};
export interface EraseShape {
    readonly kind: 'rotated-rect';
    readonly cx: number;
    readonly cy: number;
    readonly width: number;
    readonly height: number;
    readonly rotationDeg: number;
    readonly radius: number;
    readonly fill: string;
}
export interface BuildErasePlanOptions {
    readonly strategy?: EraseStrategy;
    readonly placementMargins?: readonly SafeMarginsDebug[];
}
export declare function buildErasePlan(placements: readonly TextPlacement[], options?: BuildErasePlanOptions): ErasePlan;
