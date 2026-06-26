import { ModelLoader } from '../models/ModelLoader'
import { ModelRegistry } from '../models/ModelRegistry'
import { ModelStore } from '../models/ModelStore'
import { ModelIndexedDBCache } from './ModelIndexedDBCache'
import { huggingFaceManifestUrl, proxyHuggingFaceUrl, type HuggingFaceModelRepositoryOptions } from './huggingFace'
import type { ModelDescriptor, ModelManifest } from '../domain/model'
import type { ModelAssetCache, ModelRepositoryOptions } from './modelTypes'

export class ModelRepository {
  private manifestValue: ModelManifest | null = null
  private manifestPromise: Promise<ModelManifest> | null = null
  private cacheValue: ModelAssetCache | null = null
  private pruned = false
  private readonly options: ModelRepositoryOptions
  private readonly loaders = new Map<string, ModelLoader>()

  static fromHuggingFace(options: HuggingFaceModelRepositoryOptions & {
    readonly cacheName?: string
    readonly cache?: ModelAssetCache
  }): ModelRepository {
    return new ModelRepository({
      manifestUrl: huggingFaceManifestUrl(options),
      cacheName: options.cacheName,
      cache: options.cache,
      resolveUrl: options.proxyBase ? url => proxyHuggingFaceUrl(url, options.proxyBase) : undefined,
    })
  }

  constructor(options: ModelRepositoryOptions) {
    this.options = options
    if (options.manifest) this.manifestValue = this.withResolvedUrls(options.manifest)
  }

  async model(id: string): Promise<ModelLoader> {
    let loader = this.loaders.get(id)
    if (!loader) {
      const manifest = await this.resolveManifest()
      loader = new ModelLoader(id, new ModelRegistry(manifest), new ModelStore(this.cache()))
      this.loaders.set(id, loader)
    }
    return loader
  }

  async resolveManifest(signal?: AbortSignal): Promise<ModelManifest> {
    const manifest = await this.loadOrReuseManifest(signal)
    // Best-effort, fire-and-forget: drop bytes of model versions no longer in
    // the manifest so old revisions don't accumulate in IndexedDB forever.
    void this.pruneStaleModels(manifest)
    return manifest
  }

  private cache(): ModelAssetCache {
    return (this.cacheValue ??= this.options.cache ?? new ModelIndexedDBCache())
  }

  private async loadOrReuseManifest(signal?: AbortSignal): Promise<ModelManifest> {
    if (this.manifestValue) return this.manifestValue
    if (!this.options.manifestUrl) throw new Error('model manifestUrl or manifest is required')

    if (signal) return this.loadManifest(signal)

    this.manifestPromise ??= this.loadManifest().catch(error => {
      this.manifestPromise = null
      throw error
    })
    return this.manifestPromise
  }

  private async pruneStaleModels(manifest: ModelManifest): Promise<void> {
    if (this.pruned) return
    this.pruned = true
    try {
      await new ModelStore(this.cache()).prune(Object.values(manifest.models))
    } catch {
      // GC is best-effort; never let it surface to model loading.
    }
  }

  private async loadManifest(signal?: AbortSignal): Promise<ModelManifest> {
    if (!this.options.manifestUrl) throw new Error('model manifestUrl or manifest is required')
    const manifest = this.withResolvedUrls(await fetchManifest(this.options.manifestUrl, signal))
    throwIfAborted(signal)
    this.manifestValue = manifest
    return manifest
  }

  private withResolvedUrls(manifest: ModelManifest): ModelManifest {
    if (!this.options.resolveUrl) return manifest

    let changed = false
    const models: ModelManifest['models'] = {}
    for (const [key, model] of Object.entries(manifest.models)) {
      const url = this.options.resolveUrl(model.url)
      changed ||= url !== model.url
      models[key] = url === model.url ? model : { ...model, url }
    }

    return changed ? { ...manifest, models } : manifest
  }
}

async function fetchManifest(url: string, signal?: AbortSignal): Promise<ModelManifest> {
  const res = await fetch(url, { signal })
  if (!res.ok) throw new Error(`model manifest fetch failed: ${res.status}`)
  const raw = await res.json() as unknown
  throwIfAborted(signal)
  assertValidManifest(raw)
  return raw
}

/**
 * Validate the remotely-fetched manifest before trusting it. A bare `as` cast
 * would let a malformed manifest through — most dangerously one with an empty or
 * missing sha256, which becomes an ambiguous cache key (every model collides on
 * the same IndexedDB slot). Fail loudly instead.
 */
function assertValidManifest(value: unknown): asserts value is ModelManifest {
  if (!value || typeof value !== 'object') throw new Error('model manifest is not an object')
  const models = (value as { models?: unknown }).models
  if (!models || typeof models !== 'object') throw new Error('model manifest has no models')
  const entries = Object.entries(models as Record<string, unknown>)
  if (entries.length === 0) throw new Error('model manifest has no models')
  for (const [id, model] of entries) {
    if (!model || typeof model !== 'object') throw new Error(`model manifest entry "${id}" is malformed`)
    const d = model as Partial<ModelDescriptor>
    if (typeof d.sha256 !== 'string' || d.sha256.length === 0) throw new Error(`model manifest entry "${id}" has no sha256`)
    if (typeof d.url !== 'string' || d.url.length === 0) throw new Error(`model manifest entry "${id}" has no url`)
    if (typeof d.sizeBytes !== 'number' || !(d.sizeBytes > 0)) throw new Error(`model manifest entry "${id}" has invalid sizeBytes`)
  }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

export type { ModelAssetCache, ModelDescriptor, ModelManifest, ModelRepositoryOptions, StoredModel } from './modelTypes'
