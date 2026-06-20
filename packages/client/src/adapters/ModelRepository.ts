import { ModelLoader } from '../models/ModelLoader'
import { ModelRegistry } from '../models/ModelRegistry'
import { ModelStore } from '../models/ModelStore'
import { ModelIndexedDBCache } from './ModelIndexedDBCache'
import { huggingFaceManifestUrl, proxyHuggingFaceUrl, type HuggingFaceModelRepositoryOptions } from './huggingFace'
import type { ModelManifest } from '../domain/model'
import type { ModelAssetCache, ModelRepositoryOptions } from './modelTypes'

export class ModelRepository {
  private manifestValue: ModelManifest | null = null
  private manifestPromise: Promise<ModelManifest> | null = null
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
      const cache = this.options.cache ?? new ModelIndexedDBCache()
      loader = new ModelLoader(id, new ModelRegistry(manifest), new ModelStore(cache))
      this.loaders.set(id, loader)
    }
    return loader
  }

  async resolveManifest(signal?: AbortSignal): Promise<ModelManifest> {
    if (this.manifestValue) return this.manifestValue
    if (!this.options.manifestUrl) throw new Error('model manifestUrl or manifest is required')

    if (signal) return this.loadManifest(signal)

    this.manifestPromise ??= this.loadManifest().catch(error => {
      this.manifestPromise = null
      throw error
    })
    return this.manifestPromise
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
  const manifest = await res.json() as ModelManifest
  throwIfAborted(signal)
  return manifest
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('operation aborted')
}

export type { ModelAssetCache, ModelDescriptor, ModelManifest, ModelRepositoryOptions, StoredModel } from './modelTypes'
