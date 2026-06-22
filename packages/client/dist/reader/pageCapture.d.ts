import type { EncodedOcrImage } from '../recognizers/text';
import type { ImagePixels } from '../domain/image';
import type { PageScanUnit } from '../domain/pageScan';
import type { PageSize } from '../domain/source';
import type { LoadedPage } from './pageProvider';
import type { ScanConfig } from './translationConfig';
export interface CapturedPageScan {
    readonly encoded: EncodedOcrImage;
    readonly image: ImagePixels;
    readonly captureScale: number;
    readonly haloTopPx: number;
    readonly source: PageSize;
}
export declare function capturePageScan(unit: PageScanUnit, loadPage: (index: number) => Promise<LoadedPage>, config: ScanConfig, signal: AbortSignal): Promise<CapturedPageScan>;
