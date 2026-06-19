import type { ImagePixels } from '../domain/image';
export type ImageInput = HTMLImageElement | HTMLCanvasElement | ImageBitmap | Blob | File | string;
export declare function readImageInput(input: ImageInput): Promise<ImagePixels>;
