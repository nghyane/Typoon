export declare const COMIC_DETR_INPUT_SIZE = 640;
export declare const COMIC_DETR_DEFAULT_CONFIDENCE = 0.3;
export type ComicDetrProvider = 'webgpu' | 'wasm';
export type OrtWasmPaths = string | {
    wasm?: string | URL;
    mjs?: string | URL;
};
