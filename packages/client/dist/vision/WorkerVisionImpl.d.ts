import type { ImagePixels } from '../domain/image';
import type { PreparedPageHandle } from '../domain/prepared';
import type { PageAsset } from '../domain/source';
import type { TextRegion } from '../domain/regions';
import type { PreparationStrategy } from '../domain/run';
import type { TextPlacement } from '../domain/planning';
import type { SafeMarginsDebug } from '../render/backgroundFit';
import type { EncodedOcrImage } from '../recognizers/text';
export declare class WorkerVisionImpl {
    private readonly runs;
    private readonly backend;
    beginPreparation(runId: string, strategy: PreparationStrategy): Promise<void>;
    pushPreparation(runId: string, asset: PageAsset): Promise<readonly PreparedPageHandle[]>;
    flushPreparation(runId: string): Promise<readonly PreparedPageHandle[]>;
    disposePreparation(runId: string): void;
    readPixels(runId: string, preparedPageId: string): Promise<ImagePixels>;
    encodeForOcr(runId: string, preparedPageId: string): Promise<EncodedOcrImage>;
    estimateMargins(runId: string, preparedPageId: string, placements: readonly TextPlacement[]): Promise<readonly SafeMarginsDebug[]>;
    createSeamRepair(runId: string, topPreparedPageId: string, bottomPreparedPageId: string, bandPx: number): Promise<PreparedPageHandle | null>;
    detectTextRegions(_runId: string, _preparedPageId: string): Promise<readonly TextRegion[]>;
    release(runId: string, preparedPageId: string): void;
    cancelRun(runId: string): void;
    dispose(): void;
}
