import type { ImageInput } from '../image/input';
import type { PageProjection } from './prepared';
import type { ImagePixels } from './image';
export interface PageSource {
    readonly pageCount: number;
    loadPage(index: number, signal?: AbortSignal): ImageInput | Promise<ImageInput>;
}
export interface PageSize {
    readonly width: number;
    readonly height: number;
}
export interface Rect {
    readonly x: number;
    readonly y: number;
    readonly width: number;
    readonly height: number;
}
export interface PageAsset {
    readonly index: number;
    readonly blob?: Blob;
    readonly pixels?: ImagePixels;
    readonly size?: PageSize;
    readonly projections?: readonly PageProjection[];
}
export interface PageDocumentSource {
    readonly pageCount: number;
    readPage(index: number, signal?: AbortSignal): Promise<PageAsset>;
}
