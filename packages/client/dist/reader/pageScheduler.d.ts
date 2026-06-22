import type { PageScanUnit } from '../domain/pageScan';
import type { MeasuredPage } from './pageScan';
export type PageStatus = {
    readonly kind: 'pending';
} | {
    readonly kind: 'processing';
} | {
    readonly kind: 'done';
} | {
    readonly kind: 'failed';
    readonly attempts: number;
    readonly error: string;
};
export interface VisibleRange {
    readonly top: number;
    readonly bottom: number;
    readonly center: number;
}
export declare class PageScheduler {
    private status;
    /** Reset to the latest unit set, preserving status by page index. */
    reset(units: readonly PageScanUnit[]): void;
    markProcessing(pageIndex: number): void;
    markDone(pageIndex: number): void;
    markFailed(pageIndex: number, error: string): number;
    /** Pick the nearest-to-viewport page that is pending or retryable. */
    next(units: readonly PageScanUnit[], measured: ReadonlyMap<number, MeasuredPage>, visible: VisibleRange, maxAttempts: number): PageScanUnit | null;
    progress(): {
        done: number;
        total: number;
        failed: number;
    };
    isComplete(maxAttempts: number): boolean;
    clear(): void;
}
