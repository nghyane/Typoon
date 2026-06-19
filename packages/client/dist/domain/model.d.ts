export interface ModelDescriptor {
    readonly id: string;
    readonly version: string;
    readonly url: string;
    readonly sha256: string;
    readonly sizeBytes: number;
    readonly inputSize: number;
    readonly executionProviders: readonly string[];
}
export interface ModelManifest {
    readonly version: string;
    readonly models: Record<string, ModelDescriptor>;
}
