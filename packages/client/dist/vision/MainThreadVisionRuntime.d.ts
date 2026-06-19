/**
 * Vision runtime that runs everything on the main thread.
 *
 * Used as fallback when OffscreenCanvas is unavailable, or in dev / simple
 * setups where adding a worker is premature.
 */
import type { ImagePixels } from '../domain/image';
import type { PreparedPageHandle } from '../domain/prepared';
import type { TextRegion } from '../domain/regions';
import type { TextRegionDetector } from '../detectors/textRegions';
import type { PreparationStrategy } from '../domain/run';
import type { TextPlacement } from '../domain/planning';
import type { EncodedOcrImage } from '../recognizers/text';
import type { SafeMarginsDebug } from '../render/backgroundFit';
import type { PreparationSession, VisionRuntime } from './VisionRuntime';
export declare class MainThreadVisionRuntime implements VisionRuntime {
    private readonly runs;
    private readonly deps;
    constructor(deps: {
        detector?: TextRegionDetector;
    });
    beginPreparation(runId: string, strategy: PreparationStrategy): Promise<PreparationSession>;
    readPixels(handle: PreparedPageHandle): Promise<ImagePixels>;
    encodeForOcr(handle: PreparedPageHandle, signal?: AbortSignal): Promise<EncodedOcrImage>;
    estimateMargins(handle: PreparedPageHandle, placements: readonly TextPlacement[]): Promise<readonly SafeMarginsDebug[]>;
    createSeamRepair(top: PreparedPageHandle, bottom: PreparedPageHandle, bandPx: number): Promise<PreparedPageHandle | null>;
    detectTextRegions(handle: PreparedPageHandle, signal?: AbortSignal): Promise<readonly TextRegion[]>;
    release(handle: PreparedPageHandle): void;
    cancelRun(runId: string): void;
    dispose(): void;
    private requireRun;
    private requirePage;
}
