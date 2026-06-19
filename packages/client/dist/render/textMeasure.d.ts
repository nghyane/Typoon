import type { FontProfile } from './font';
export interface MeasureRequest {
    readonly text: string;
    readonly width: number;
    readonly height: number;
    readonly fontSizePx: number;
    readonly fontWeight: string;
}
export interface BlockMeasureRequest {
    readonly text: string;
    readonly width: number;
    readonly fontSizePx: number;
    readonly fontWeight: string;
}
export interface TextWidthRequest {
    readonly text: string;
    readonly fontSizePx: number;
    readonly fontWeight: string;
}
export interface TextBlockMeasure {
    readonly widthPx: number;
    readonly heightPx: number;
    readonly lineCount: number;
    readonly overflowWidth: boolean;
}
export interface DomMeasurer {
    fits(request: MeasureRequest): boolean;
    measure(request: BlockMeasureRequest): TextBlockMeasure;
    textWidth(request: TextWidthRequest): number;
    destroy(): void;
}
export declare function createDomMeasurer(font: FontProfile): DomMeasurer;
export declare function maxFittingSize(args: {
    readonly text: string;
    readonly width: number;
    readonly height: number;
    readonly hiBound: number;
    readonly fontWeight: string;
    readonly measurer: DomMeasurer;
    readonly minFontSize: number;
}): number;
