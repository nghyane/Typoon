import type { TextPlacement } from '../domain/planning';
export interface OverlayDebugOptions {
    readonly showDrawable?: boolean;
    readonly showTextBoxes?: boolean;
    readonly showTextBounds?: boolean;
    readonly showLabels?: boolean;
}
export declare function createDebugLayer(placements: readonly TextPlacement[], pageSize: readonly [number, number], options?: OverlayDebugOptions): SVGSVGElement;
