export interface PipelineConcurrency {
    /** Max full-page pixel pipelines allowed to be live at once. */
    readonly maxPreparedPages?: number;
    readonly load: number;
    readonly prepare: number;
    readonly ocr: number;
    readonly detect: number;
    readonly translate: number;
    readonly compose: number;
}
/**
 * Concurrency + retry gate for each pipeline stage.
 *
 * Owns no page state, no progress, no queue awareness.
 * Just "how many of this stage can run at once, and should I retry?"
 */
export declare class StageExecutor {
    private readonly loadLimit;
    private readonly prepareLimit;
    private readonly ocrLimit;
    private readonly detectLimit;
    private readonly translateLimit;
    private readonly composeLimit;
    constructor(concurrency: PipelineConcurrency);
    load<T>(task: () => Promise<T>): Promise<T>;
    prepare<T>(task: () => Promise<T>): Promise<T>;
    ocr<T>(task: () => Promise<T>): Promise<T>;
    detect<T>(task: () => Promise<T>): Promise<T>;
    translate<T>(task: () => Promise<T>): Promise<T>;
    compose<T>(task: () => Promise<T>): Promise<T>;
}
