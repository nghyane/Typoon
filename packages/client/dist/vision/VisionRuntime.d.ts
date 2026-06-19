/**
 * Vision is all heavy image work: decode, prepare (identity/stitch/cut),
 * and ONNX detection.
 *
 * PreparedPageHandle is a proxy — pixel data stays on the vision side
 * and is only transferred when `readPixels()` is called.
 */
import type { PageAsset } from '../domain/source';
import type { PreparedPageHandle } from '../domain/prepared';
import type { ImagePixels } from '../domain/image';
import type { TextRegion } from '../domain/regions';
import type { PreparationStrategy } from '../domain/run';
import type { TextPlacement } from '../domain/planning';
import type { SafeMarginsDebug } from '../render/backgroundFit';
import type { EncodedOcrImage } from '../recognizers/text';
/** Session scoped to one translation run. */
export interface PreparationSession {
    /**
     * Push a loaded source page into the preparation stream.
     * May emit zero, one, or many PreparedPageHandles depending on the strategy
     * (identity → one page; stitch → may buffer; cut → may emit multiple).
     */
    push(asset: PageAsset, signal?: AbortSignal): Promise<readonly PreparedPageHandle[]>;
    /** Flush any buffered pages at the end of the source stream. */
    flush(signal?: AbortSignal): Promise<readonly PreparedPageHandle[]>;
    /** Release preparation-only buffers that were not emitted as prepared pages. */
    dispose?(): void;
}
export interface VisionRuntime {
    beginPreparation(runId: string, strategy: PreparationStrategy): Promise<PreparationSession>;
    /** Read pixel data for a prepared page. Transfer-once from worker. */
    readPixels(handle: PreparedPageHandle, signal?: AbortSignal): Promise<ImagePixels>;
    /** Encode prepared page into OCR upload bytes without exposing RGBA pixels. */
    encodeForOcr(handle: PreparedPageHandle, signal?: AbortSignal): Promise<EncodedOcrImage>;
    /** Estimate text erase/expand margins while pixels stay in the vision side. */
    estimateMargins(handle: PreparedPageHandle, placements: readonly TextPlacement[], signal?: AbortSignal): Promise<readonly SafeMarginsDebug[]>;
    /** Build a small cross-page OCR repair image from two adjacent prepared source pages. */
    createSeamRepair?(top: PreparedPageHandle, bottom: PreparedPageHandle, bandPx: number, signal?: AbortSignal): Promise<PreparedPageHandle | null>;
    /** Run ONNX detection on a prepared page (inside worker when available). */
    detectTextRegions(handle: PreparedPageHandle, signal?: AbortSignal): Promise<readonly TextRegion[]>;
    /** Free worker-side pixel memory for this handle. */
    release(handle: PreparedPageHandle): void;
    /** Cancel an entire run — clears all in-flight work and caches. */
    cancelRun(runId: string): void;
    dispose(): void;
}
