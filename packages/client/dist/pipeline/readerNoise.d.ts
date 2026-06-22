import type { BBox } from '../domain/geometry';
import type { RecognizedTextPage } from '../domain/text';
export interface NoiseFrame {
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
}
/** Drop watermark/credit/noise blocks, keeping real text. */
export declare function removeReaderNoiseBlocks(recognized: RecognizedTextPage, frame: NoiseFrame): RecognizedTextPage;
export declare function isReaderNoiseText(text: string, bbox: BBox, frame: NoiseFrame): boolean;
