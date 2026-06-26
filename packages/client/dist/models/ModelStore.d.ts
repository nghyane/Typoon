import type { ModelDescriptor } from '../domain/model';
import type { ModelAssetCache } from '../adapters/modelTypes';
export declare class ModelStore {
    private readonly cache;
    constructor(cache: ModelAssetCache);
    key(model: ModelDescriptor): string;
    read(model: ModelDescriptor): Promise<ArrayBuffer | null>;
    write(model: ModelDescriptor, bytes: ArrayBuffer): Promise<void>;
    /**
     * Delete cached model bytes whose key is not in `keep` — the garbage
     * collector for stale model versions. Without this, every model revision
     * leaves its old bytes in IndexedDB forever. Best-effort: skipped if the
     * cache backend cannot enumerate or delete keys. Returns the number pruned.
     */
    prune(keep: Iterable<ModelDescriptor>): Promise<number>;
}
