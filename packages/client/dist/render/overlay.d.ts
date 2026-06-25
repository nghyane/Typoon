import type { TextPlacement } from '../domain/planning';
import type { TranslatedUnit } from '../domain/translation';
import type { SafeMarginsDebug } from './backgroundFit';
import { type OverlayDebugOptions } from './debugLayer';
import { type EraseStrategy } from './erasePlan';
export interface OverlayOptions {
    readonly eraseStrategy?: EraseStrategy;
    readonly debug?: OverlayDebugOptions;
}
export interface OverlayRenderData {
    readonly placements: readonly TextPlacement[];
    readonly translations: readonly TranslatedUnit[];
    readonly pageSize: readonly [number, number];
    readonly placementMargins?: readonly SafeMarginsDebug[];
    readonly fontContextPlacements?: readonly TextPlacement[];
    readonly sourceLanguage?: string | null;
    readonly targetLanguage?: string | null;
}
export declare function attachOverlay(host: HTMLElement, data: OverlayRenderData, options?: OverlayOptions): HTMLElement;
export declare function createOverlayElement(data: OverlayRenderData, options?: OverlayOptions): HTMLElement;
