import { AsyncLimiter } from './AsyncLimiter';
export interface StageConcurrencyPolicy {
    readonly pages: number;
    readonly recognize: number;
    readonly detect: number;
    readonly translate: number;
}
export interface StageRetryOptions {
    readonly attempts?: number;
    readonly baseDelayMs?: number;
    readonly maxDelayMs?: number;
}
export interface StageSchedulerPolicy {
    readonly concurrency: StageConcurrencyPolicy;
    readonly retry?: {
        readonly recognize?: StageRetryOptions;
        readonly detect?: StageRetryOptions;
        readonly translate?: StageRetryOptions;
    };
}
export type StageSchedulerOptions = Partial<StageSchedulerPolicy> & {
    readonly concurrency?: Partial<StageConcurrencyPolicy>;
};
export declare class StageScheduler {
    readonly pages: AsyncLimiter;
    private readonly policy;
    private readonly recognizeLimiter;
    private readonly detectLimiter;
    private readonly translateLimiter;
    constructor(policy: StageSchedulerPolicy);
    recognize<T>(task: () => Promise<T>): Promise<T>;
    detect<T>(task: () => Promise<T>): Promise<T>;
    translate<T>(task: () => Promise<T>): Promise<T>;
}
