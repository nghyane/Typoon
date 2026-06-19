import type { ImagePixels } from '../domain/image';
import type { PreparedPageHandle } from '../domain/prepared';
import type { TextRegion } from '../domain/regions';
import type { PreparationStrategy } from '../domain/run';
import type { TextPlacement } from '../domain/planning';
import type { SafeMarginsDebug } from '../render/backgroundFit';
import type { EncodedOcrImage } from '../recognizers/text';
import type { TextRegionDetector } from '../detectors/textRegions';
import type { PreparationSession, VisionRuntime } from './VisionRuntime';
export declare class ComlinkVisionRuntime implements VisionRuntime {
    private readonly proxy;
    private readonly worker;
    private readonly detector;
    private workerError;
    private disposed;
    constructor(worker: Worker, options?: {
        readonly detector?: TextRegionDetector;
    });
    beginPreparation(runId: string, strategy: PreparationStrategy): Promise<PreparationSession>;
    readPixels(handle: PreparedPageHandle, _signal?: AbortSignal): Promise<ImagePixels>;
    encodeForOcr(handle: PreparedPageHandle, _signal?: AbortSignal): Promise<EncodedOcrImage>;
    estimateMargins(handle: PreparedPageHandle, placements: readonly TextPlacement[], _signal?: AbortSignal): Promise<readonly SafeMarginsDebug[]>;
    createSeamRepair(top: PreparedPageHandle, bottom: PreparedPageHandle, bandPx: number, _signal?: AbortSignal): Promise<PreparedPageHandle | null>;
    detectTextRegions(handle: PreparedPageHandle, signal?: AbortSignal): Promise<readonly TextRegion[]>;
    release(handle: PreparedPageHandle): void;
    cancelRun(runId: string): void;
    dispose(): void;
    private callWorker;
}
