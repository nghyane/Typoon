import type { ModelDescriptor, ModelManifest } from '../domain/model';
export type { ModelDescriptor, ModelManifest };
export interface ModelRepositoryOptions {
    readonly manifestUrl?: string;
    readonly manifest?: ModelManifest;
    readonly cacheName?: string;
    readonly cache?: ModelAssetCache;
    readonly resolveUrl?: (url: string) => string;
}
export interface StoredModel {
    readonly id: string;
    readonly descriptor: ModelDescriptor;
    readonly bytes: ArrayBuffer;
}
export interface ModelAssetCache {
    match(key: string): Promise<ArrayBuffer | null>;
    put(key: string, bytes: ArrayBuffer): Promise<void>;
}
