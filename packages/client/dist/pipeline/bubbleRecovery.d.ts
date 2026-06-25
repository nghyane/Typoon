import type { ImagePixels } from '../domain/image';
import type { RecognizedTextPage } from '../domain/text';
import type { TextRegion } from '../domain/regions';
export interface BubbleCropRecognizer {
    recognizeCrop(image: ImagePixels): Promise<RecognizedTextPage>;
}
/** Coordinate mapping + source-image access for Phase B crops. */
export interface BubbleSource {
    /** Load the full-resolution stitched canvas (page N + both halos) at 1:1,
     *  shared across all anchor crops for this page.  Origin matches the capture
     *  canvas, so capture-space bbox → source-space = bbox / captureScale. */
    readonly loadFullCanvas: () => Promise<HTMLCanvasElement>;
    /** Scale factor: source px → capture px. */
    readonly captureScale: number;
}
/**
 * Re-OCR incomplete bubbles and splice the recovered text into `recognized`.
 * Returns the original page unchanged when there are no regions/anchors or
 * nothing needs recovery.
 */
export declare function recoverBubbleText(args: {
    readonly recognized: RecognizedTextPage;
    readonly source: BubbleSource;
    readonly regions: readonly TextRegion[];
    readonly recognizer: BubbleCropRecognizer;
}): Promise<RecognizedTextPage>;
