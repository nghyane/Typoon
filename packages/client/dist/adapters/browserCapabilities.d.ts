export interface BrowserModelHint {
    /** Model ID to use from the manifest. */
    readonly modelId: string;
    /** Provider to prefer when creating the ORT session. */
    readonly preferredProvider: 'webgpu' | 'wasm';
    /** Number of wasm threads (1 for Safari / environments without SharedArrayBuffer). */
    readonly wasmNumThreads: number;
}
export interface BrowserCapabilities {
    readonly supportsWebGpu: boolean;
    readonly supportsStableWebGpu: boolean;
    readonly isSafari: boolean;
    readonly isIOS: boolean;
    readonly isMobile: boolean;
    readonly modelHint: BrowserModelHint;
}
/** Detect browser capabilities and return the optimal model hint. */
export declare function detectBrowserCapabilities(): BrowserCapabilities;
