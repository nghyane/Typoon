import { ModelAsset } from './ModelAsset'
import type { ModelManifest, ModelRepositoryOptions } from './modelTypes'

export class ModelRepository {
  private manifestValue: ModelManifest | null = null
  private manifestPromise: Promise<ModelManifest> | null = null
  private readonly assets = new Map<string, ModelAsset>()

  constructor(private readonly options: ModelRepositoryOptions) {}

  model(id: string): ModelAsset {
    let asset = this.assets.get(id)
    if (!asset) {
      asset = new ModelAsset(id, signal => this.loadManifest(signal), this.options)
      this.assets.set(id, asset)
    }
    return asset
  }

  private async loadManifest(signal?: AbortSignal): Promise<ModelManifest> {
    if (this.options.manifest) return this.options.manifest
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
