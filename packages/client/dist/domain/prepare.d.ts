import type { PageDocumentSource } from './source';
export type SourcePresentation = 'paged' | 'continuous-strip' | 'unknown';
export type PrepareStrategy = {
    readonly type: 'identity';
} | {
    readonly type: 'identity-with-seams';
    readonly seamBandPx?: number;
} | {
    readonly type: 'continuous-strip';
};
export interface PrepareRequest {
    readonly runId: string;
    readonly source: PageDocumentSource;
    readonly strategy: PrepareStrategy;
    readonly profile: PrepareProfile;
    readonly artifacts?: PrepareArtifactSink;
    readonly signal?: AbortSignal;
}
export interface PrepareProfile {
    readonly sourcePresentation: SourcePresentation;
    readonly seam: SeamPolicy;
    readonly memory: PrepareMemoryPolicy;
}
export interface SeamPolicy {
    readonly bandFraction: number;
    readonly minBandPx: number;
    readonly maxBandPx: number;
    readonly previewWidthPx: number;
    readonly decision: SeamDecisionPolicy;
}
export interface SeamDecisionPolicy {
    readonly mergeConfidence: number;
    readonly cutConfidence: number;
    readonly uncertainAction: 'cut' | 'merge';
}
export interface PrepareMemoryPolicy {
    readonly maxMergePages: number;
    readonly maxPreparedHeightPx: number;
    readonly maxLiveDecodedPages: number;
}
export interface PrepareArtifactSink {
    writeJson(path: string, data: unknown): Promise<void>;
    writeImage(path: string, image: import('./image').ImagePixels): Promise<void>;
}
export declare const DEFAULT_PREPARE_PROFILE: PrepareProfile;
export declare const CONTINUOUS_STRIP_PREPARE_PROFILE: PrepareProfile;
