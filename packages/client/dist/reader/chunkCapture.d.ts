import type { ChapterContentLayout } from '../domain/chapterContent';
import type { SourcePageSize } from '../pipeline/chapterContent';
export interface VisibleContentRange {
    readonly top: number;
    readonly bottom: number;
    readonly center: number;
}
/** Measure the live reader DOM into a chapter content layout. */
export declare function measureLayout(host: HTMLElement | null, knownSize: (pageIndex: number) => SourcePageSize | null): ChapterContentLayout | null;
export declare function visibleContentRange(host: HTMLElement, contentSize: {
    readonly width: number;
    readonly height: number;
}, marginPx: number): VisibleContentRange;
