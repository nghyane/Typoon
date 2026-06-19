import type { ImagePixels } from '../../../domain/image';
export interface SignalScore {
    readonly score: number;
    readonly confidence: number;
}
export interface SeamSignals {
    readonly bubbleComponentCrossing: SignalScore;
    readonly textInkCrossing: SignalScore;
    readonly panelGutter: SignalScore;
    readonly edgeContinuity: SignalScore;
}
export declare function analyzeSeamSignals(args: {
    readonly topBand: ImagePixels;
    readonly bottomBand: ImagePixels;
}): SeamSignals;
