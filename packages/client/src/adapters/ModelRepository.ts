import { ModelLoader } from '../models/ModelLoader'
import { ModelRegistry } from '../models/ModelRegistry'
import { ModelStore } from '../models/ModelStore'
import { ModelIndexedDBCache } from './ModelIndexedDBCache'
import { huggingFaceManifestUrl, type HuggingFaceModelRepositoryOptions } from './huggingFace'
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
    })
  }

  constructor(options: ModelRepositoryOptions) {
    this.options = options
    if (options.manifest) this.manifestValue = options.manifest
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
    if (this.options.manifest) {
      this.manifestValue = this.options.manifest
      return this.options.manifest
    }
    if (this.manifestValue) return this.manifestValue
    if (!this.options.manifestUrl) throw new Error('model manifestUrl or manifest is required')

    if (signal) {
      const manifest = await fetchManifest(this.options.manifestUrl, signal)
      throwIfAborted(signal)
      this.manifestValue = manifest
      return manifest
    }

    this.manifestPromise ??= fetchManifest(this.options.manifestUrl).then(manifest => {
      this.manifestValue = manifest
      return manifest
    }).catch(error => {
      this.manifestPromise = null
      throw error
    })
    return this.manifestPromise
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
