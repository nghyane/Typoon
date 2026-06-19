import type { PrepareProfile } from '../../../domain/prepare';
import type { ImagePixels } from '../../../domain/image';
import type { CanvasBackend } from '../canvasBackend';
import { type SeamSignals } from './signals';
import { type SeamDecision } from './decision';
export interface SeamAnalysis {
    readonly boundary: {
        readonly topSourcePageIndex: number;
        readonly bottomSourcePageIndex: number;
    };
    readonly decision: SeamDecision;
    readonly signals: SeamSignals;
    readonly bandPx: number;
}
export declare function detectSeam(args: {
    readonly topSourcePageIndex: number;
    readonly bottomSourcePageIndex: number;
    readonly top: ImagePixels;
    readonly bottom: ImagePixels;
    readonly profile: PrepareProfile;
    readonly backend: CanvasBackend;
    readonly signal?: AbortSignal;
}): Promise<SeamAnalysis>;
