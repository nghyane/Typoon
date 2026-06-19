import type { ImagePixels } from '../../domain/image';
export interface DownsampleBandArgs {
    readonly image: ImagePixels;
    readonly edge: 'top' | 'bottom';
    readonly bandPx: number;
    readonly targetWidthPx: number;
    readonly signal?: AbortSignal;
}
export interface CanvasBackend {
    decode(blob: Blob, signal?: AbortSignal): Promise<ImagePixels>;
    stitchVertical(images: readonly ImagePixels[], signal?: AbortSignal): Promise<ImagePixels>;
    encodePng(image: ImagePixels, signal?: AbortSignal): Promise<Blob>;
    downsampleBand(args: DownsampleBandArgs): Promise<ImagePixels>;
}
export declare class BrowserCanvasBackend implements CanvasBackend {
    decode(blob: Blob, signal?: AbortSignal): Promise<ImagePixels>;
    stitchVertical(images: readonly ImagePixels[], signal?: AbortSignal): Promise<ImagePixels>;
    encodePng(image: ImagePixels, signal?: AbortSignal): Promise<Blob>;
    downsampleBand(args: DownsampleBandArgs): Promise<ImagePixels>;
}
