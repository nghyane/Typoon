import type { ModelDescriptor } from '../domain/model'
import type { ModelAssetCache } from '../adapters/modelTypes'

export class ModelStore {
  private readonly cache: ModelAssetCache

  constructor(cache: ModelAssetCache) {
    this.cache = cache
  }

  key(model: ModelDescriptor): string {
    // sha256 is the sole cache key. An empty key would collapse every model
    // onto the same slot — they'd overwrite each other and reads would return
    // the wrong bytes. Refuse rather than corrupt the cache. Manifest
    // validation should make this unreachable in practice.
    if (!model.sha256) throw new Error(`model "${model.id}" has no sha256; refusing to cache by ambiguous key`)
    return model.sha256
  }

  async read(model: ModelDescriptor): Promise<ArrayBuffer | null> {
    return this.cache.match(this.key(model))
  }

  async write(model: ModelDescriptor, bytes: ArrayBuffer): Promise<void> {
    return this.cache.put(this.key(model), bytes)
  }

  /**
   * Delete cached model bytes whose key is not in `keep` — the garbage
   * collector for stale model versions. Without this, every model revision
   * leaves its old bytes in IndexedDB forever. Best-effort: skipped if the
   * cache backend cannot enumerate or delete keys. Returns the number pruned.
   */
  async prune(keep: Iterable<ModelDescriptor>): Promise<number> {
    if (!this.cache.keys || !this.cache.delete) return 0
    const keepKeys = new Set<string>()
    for (const model of keep) keepKeys.add(this.key(model))
    const existing = await this.cache.keys()
    const stale = existing.filter(key => !keepKeys.has(key))
    await Promise.all(stale.map(key => this.cache.delete!(key)))
    return stale.length
  }
}
