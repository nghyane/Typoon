import type { ModelDescriptor, ModelManifest } from '../domain/model';
export declare class ModelRegistry {
    private readonly manifest;
    constructor(manifest: ModelManifest);
    get(id: string): ModelDescriptor;
}
