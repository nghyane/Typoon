export interface ModelDescriptor {
  readonly url: string
  readonly sha256?: string
  readonly sizeBytes?: number
  readonly inputSize?: number
  readonly executionProviders?: readonly string[]
}

export interface ModelManifest {
  readonly version: string
  readonly models: Record<string, ModelDescriptor>
}

export interface ModelRepositoryOptions {
  readonly manifestUrl?: string
  readonly manifest?: ModelManifest
  readonly cacheName?: string
  readonly cache?: ModelAssetCache
}

export interface StoredModel {
  readonly id: string
  readonly descriptor: ModelDescriptor
  readonly bytes: ArrayBuffer
}

export interface ModelAssetCache {
  match(key: string): Promise<ArrayBuffer | null>
  put(key: string, bytes: ArrayBuffer): Promise<void>
}
