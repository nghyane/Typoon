import type * as ort from 'onnxruntime-web/wasm';
import type { OrtModule } from './OrtBackend';
export type OrtProvider = 'webgpu' | 'wasm';
export interface OrtSessionHandle {
    readonly ort: OrtModule;
    readonly session: ort.InferenceSession;
    readonly provider: OrtProvider;
    readonly descriptorId: string;
}
export declare class OrtSessionPool {
    private readonly ort;
    private readonly sessions;
    constructor(ort: OrtModule);
    session(descriptorId: string, bytes: ArrayBuffer, providers: readonly string[]): Promise<OrtSessionHandle>;
}
