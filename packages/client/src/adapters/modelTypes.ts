import type { ModelDescriptor, ModelManifest } from '../domain/model'

export type { ModelDescriptor, ModelManifest }

export interface ModelRepositoryOptions {
  readonly manifestUrl?: string
  readonly manifest?: ModelManifest
  readonly cacheName?: string
  readonly cache?: ModelAssetCache
  readonly resolveUrl?: (url: string) => string
}

export interface StoredModel {
  readonly id: string
  readonly descriptor: ModelDescriptor
  readonly bytes: ArrayBuffer
}

export interface ModelAssetCache {
  match(key: string): Promise<ArrayBuffer | null>
  put(key: string, bytes: ArrayBuffer): Promise<void>
  /** Enumerate all stored keys. Optional: enables stale-model garbage collection. */
  keys?(): Promise<string[]>
  /** Remove a stored entry by key. Optional: enables stale-model garbage collection. */
  delete?(key: string): Promise<void>
}
