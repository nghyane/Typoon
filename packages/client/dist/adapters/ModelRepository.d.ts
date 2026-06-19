import { ModelLoader } from '../models/ModelLoader';
import { type HuggingFaceModelRepositoryOptions } from './huggingFace';
import type { ModelManifest } from '../domain/model';
import type { ModelAssetCache, ModelRepositoryOptions } from './modelTypes';
export declare class ModelRepository {
    private manifestValue;
    private manifestPromise;
    private readonly options;
    private readonly loaders;
    static fromHuggingFace(options: HuggingFaceModelRepositoryOptions & {
        readonly cacheName?: string;
        readonly cache?: ModelAssetCache;
    }): ModelRepository;
    constructor(options: ModelRepositoryOptions);
    model(id: string): Promise<ModelLoader>;
    resolveManifest(signal?: AbortSignal): Promise<ModelManifest>;
}
export type { ModelAssetCache, ModelDescriptor, ModelManifest, ModelRepositoryOptions, StoredModel } from './modelTypes';
