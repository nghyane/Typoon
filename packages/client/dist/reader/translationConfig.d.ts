import type { OrtProvider } from '../models/OrtSessionPool';
export interface TranslationConfig {
    readonly model: {
        readonly repo: string;
        readonly revision: string;
        readonly proxyBase: string;
    };
    readonly chunk: {
        readonly overlayMarginPx: number;
        readonly processMarginPx: number;
    };
    readonly scan: ScanConfig;
    readonly resilience: {
        readonly maxChunkAttempts: number;
        readonly backoffMs: number;
    };
    readonly memory: {
        readonly maxCachedPages: number;
    };
    readonly translator: {
        readonly maxSessionsDesktop: number;
        readonly maxSessionsMobile: number;
    };
}
export interface ScanConfig {
    /** Cap OCR canvas width (source px) to bound recognition/detection cost. */
    readonly maxCaptureWidth: number;
    /** Halo height as a fraction of the page's source height. */
    readonly haloRatio: number;
    /** Absolute cap on halo height (source px). */
    readonly haloMaxPx: number;
}
export declare const defaultTranslationConfig: TranslationConfig;
export declare function preferredProviders(preferred: OrtProvider): readonly OrtProvider[];
