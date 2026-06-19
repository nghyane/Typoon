import type { ModelAssetCache } from './modelTypes';
export declare class ModelIndexedDBCache implements ModelAssetCache {
    private dbPromise;
    private db;
    match(key: string): Promise<ArrayBuffer | null>;
    put(key: string, bytes: ArrayBuffer): Promise<void>;
}
