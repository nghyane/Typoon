import { ModelLoader } from '../models/ModelLoader'
import { ModelRegistry } from '../models/ModelRegistry'
import { ModelStore } from '../models/ModelStore'
import { BrowserModelCache } from './BrowserModelCache'
import type { ModelManifest } from '../domain/model'
import type { ModelRepositoryOptions } from './modelTypes'

export class ModelRepository {
  private manifestValue: ModelManifest | null = null
  private manifestPromise: Promise<ModelManifest> | null = null
  private readonly loaders = new Map<string, ModelLoader>()

  constructor(private readonly options: ModelRepositoryOptions) {
    if (options.manifest) this.manifestValue = options.manifest
  }

  model(id: string): ModelLoader {
    let loader = this.loaders.get(id)
    if (!loader) {
      const manifest = this.manifestValue
      if (!manifest) throw new Error('model manifest not loaded; call resolveManifest() or pass manifest in options')
      const cache = this.options.cache ?? new BrowserModelCache(
        this.options.cacheName ?? `typoon-models-${manifest.version}`,
      )
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
