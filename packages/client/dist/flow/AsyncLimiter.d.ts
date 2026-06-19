export declare class AsyncLimiter {
    readonly concurrency: number;
    private active;
    private readonly queue;
    constructor(concurrency: number);
    run<T>(task: () => Promise<T>): Promise<T>;
    private acquire;
    private release;
}
