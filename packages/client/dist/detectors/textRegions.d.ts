import type { ImagePixels } from '../domain/image';
import type { TextRegion } from '../domain/regions';
export type { TextRegion, TextRegionKind } from '../domain/regions';
export interface TextRegionDetector {
    readonly name: string;
    detectTextRegions(image: ImagePixels, options?: {
        readonly signal?: AbortSignal;
    }): Promise<readonly TextRegion[]>;
}
