import type { TextPlacement } from '../domain/planning';
import type { TranslatedUnit } from '../domain/translation';
import { type SafeMarginsDebug } from './backgroundFit';
import { type CssFitResult } from './fit';
import type { RenderLanguageContext } from './languageProfile';
export interface TextLayerItem {
    readonly placement: TextPlacement;
    readonly unit: TranslatedUnit;
}
export interface TextLayerOptions {
    readonly placementMargins?: readonly SafeMarginsDebug[];
    readonly fontContextPlacements?: readonly TextPlacement[];
    readonly languageContext?: RenderLanguageContext;
}
export type FittedTextLayerItem<T extends TextLayerItem = TextLayerItem> = T & {
    readonly fit: CssFitResult;
};
export declare function fitTextLayerItems<T extends TextLayerItem>(items: readonly T[], pageSize: readonly [number, number], options?: TextLayerOptions): Array<FittedTextLayerItem<T>>;
export declare function createTextLayer(items: readonly TextLayerItem[], pageSize: readonly [number, number], options?: TextLayerOptions): HTMLElement;
export declare function createTextLayerFromFits(items: readonly FittedTextLayerItem[], pageSize: readonly [number, number]): HTMLElement;
