import type { ImagePixels } from '../domain/image';
export interface TextPresenceResult {
    readonly hasText: boolean;
    /** Fraction of grid blocks that contain text-like texture. */
    readonly textBlockFraction: number;
}
export declare function detectTextPresence(image: ImagePixels): TextPresenceResult;
