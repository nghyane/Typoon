import type { ModelDescriptor, ModelManifest } from '../domain/model'

export class ModelRegistry {
  private readonly manifest: ModelManifest

  constructor(manifest: ModelManifest) {
    this.manifest = manifest
  }

  get(id: string): ModelDescriptor {
    const model = this.manifest.models[id]
    if (!model) throw new Error(`model not found in manifest: ${id}`)
    return model
  }
}
