/** Throw signal.reason (or a generic Error) if the signal is already aborted. */
export declare function throwIfAborted(signal: AbortSignal): void;
/** Normalize any thrown value to a message string. */
export declare function errorMessage(error: unknown): string;
/** A signal that never aborts — for background work that must outlive a run. */
export declare function neverAbort(): AbortSignal;
/** Run once the main thread is idle (requestIdleCallback, setTimeout fallback). */
export declare function whenIdle(run: () => void, timeoutMs?: number): void;
/** Promise that resolves when the main thread is idle. */
export declare function yieldToIdle(timeoutMs: number): Promise<void>;
