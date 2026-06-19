import type { ImagePixels } from '../domain/image';
import type { RecognizedTextPage } from '../domain/text';
export interface EncodedOcrImage {
    readonly bytes: Uint8Array;
    readonly width: number;
    readonly height: number;
    readonly originalWidth: number;
    readonly originalHeight: number;
    /** Fast pre-check: false = page has no text → skip Lens API call. */
    readonly hasText?: boolean;
}
export interface TextRecognitionOptions {
    readonly sourceLang: string | null;
    readonly pageIndex: number;
    readonly signal?: AbortSignal;
}
export interface TextRecognizer {
    readonly name: string;
    recognizeText(image: ImagePixels, options: TextRecognitionOptions): Promise<RecognizedTextPage>;
    recognizeEncoded?(image: EncodedOcrImage, options: TextRecognitionOptions): Promise<RecognizedTextPage>;
}
