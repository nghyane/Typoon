import type { ImagePixels } from '../domain/image';
import type { TextRegion } from '../domain/regions';
import { type TranslationConfig } from './translationConfig';
export interface ReaderModelState {
    readonly state: 'idle' | 'resolving' | 'downloading' | 'initializing' | 'ready' | 'failed';
    readonly receivedBytes?: number;
    readonly totalBytes?: number;
    readonly ratio?: number;
    readonly error?: string;
}
export declare function subscribeModelState(listener: (state: ReaderModelState) => void): () => void;
export declare function detectTextRegions(image: ImagePixels, signal: AbortSignal, config?: TranslationConfig): Promise<readonly TextRegion[]>;
/**
 * Warm the detector singleton in the background so the first OCR page does not
 * pay for the ORT backend import + model download + session compile. Safe to
 * call repeatedly; a no-op once a load is in flight or done. Scheduled on idle
 * so it never competes with the reader's first paint.
 */
export declare function prewarmTextRegionDetector(config?: TranslationConfig): void;
