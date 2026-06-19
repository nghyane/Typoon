import type { ImagePixels } from '../../domain/image';
import type { PreparedPageAsset } from '../../domain/preparedChapter';
import type { CanvasBackend } from './canvasBackend';
export declare class MemoryPreparedPageAsset implements PreparedPageAsset {
    private pixels;
    private blobPromise;
    private readonly backend;
    constructor(image: ImagePixels, backend: CanvasBackend);
    readPixels(signal?: AbortSignal): Promise<ImagePixels>;
    readBlob(signal?: AbortSignal): Promise<Blob>;
    release(): void;
}
