import type { ChapterContentLayout } from '../domain/chapterContent';
import type { PageScanUnit } from '../domain/pageScan';
import type { PageSize } from '../domain/source';
import type { ScanConfig } from './translationConfig';
export interface PageSource {
    readonly pageIndex: number;
    readonly source: PageSize;
}
/** Build one scan unit per page; halo is taken from the adjacent page. */
export declare function planPageScans(pages: readonly PageSource[], config: ScanConfig): readonly PageScanUnit[];
/** A measured page from the live reader DOM (for viewport ordering only). */
export interface MeasuredPage {
    readonly pageIndex: number;
    readonly source: PageSize;
    readonly domTop: number;
    readonly domHeight: number;
}
export declare function measuredPagesFromLayout(layout: ChapterContentLayout): readonly MeasuredPage[];
