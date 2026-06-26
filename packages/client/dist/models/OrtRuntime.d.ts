import type { OrtModule } from './OrtBackend';
export interface OrtRuntimeOptions {
    readonly wasmPaths?: string | {
        wasm?: string | URL;
        mjs?: string | URL;
    };
    readonly wasmNumThreads?: number;
    readonly logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal';
}
export declare class OrtRuntime {
    private readonly ort;
    private configured;
    constructor(ort: OrtModule);
    configure(options: OrtRuntimeOptions): void;
}
