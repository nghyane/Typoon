import type { TextPlacement } from '../domain/planning';
export interface TextStrokePlan {
    readonly color: string;
    readonly widthPx: number;
}
export interface TextStylePlan {
    readonly fill: string;
    readonly fontWeight: string;
    readonly strokes: readonly TextStrokePlan[];
    readonly shadow: string | null;
}
export declare function buildTextStyle(placement: TextPlacement, fontSizePx?: number): TextStylePlan;
