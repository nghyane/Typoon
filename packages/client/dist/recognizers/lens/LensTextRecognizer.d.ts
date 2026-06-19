import type { ImagePixels } from '../../domain/image';
import type { RecognizedTextPage } from '../../domain/text';
import type { EncodedOcrImage, TextRecognizer, TextRecognitionOptions } from '../text';
export interface LensTextRecognizerOptions {
    readonly endpoint?: string;
    readonly requestTimeoutMs?: number;
    readonly region?: string;
    readonly timeZone?: string;
}
export declare class LensTextRecognizer implements TextRecognizer {
    readonly name = "lens-text-recognizer";
    private readonly options;
    private readonly endpoint;
    private readonly requestTimeoutMs;
    constructor(options?: LensTextRecognizerOptions);
    recognizeText(image: ImagePixels, options: TextRecognitionOptions): Promise<RecognizedTextPage>;
    recognizeEncoded(image: EncodedOcrImage, options: TextRecognitionOptions): Promise<RecognizedTextPage>;
}
