import type { ModelLoader } from '../../models/ModelLoader';
import { type Capability, type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../../domain/capability';
import type { ImagePixels } from '../../domain/image';
import type { TextRegion } from '../../domain/regions';
import type { TextRegionDetector } from '../textRegions';
import { type ComicDetrProvider } from './ortTypes';
import type { OrtSessionPool } from '../../models/OrtSessionPool';
export interface ComicDetrDetectorOptions {
    readonly model: ModelLoader;
    readonly confidenceThreshold?: number;
    readonly preferredProviders?: readonly ComicDetrProvider[];
    readonly sessionPool: OrtSessionPool;
}
export declare class MangaTextRegionDetector implements TextRegionDetector, Capability {
    readonly name = "manga-text-region-detector";
    private readonly capability;
    private readonly runner;
    constructor(options: ComicDetrDetectorOptions);
    status(): CapabilityStatus;
    subscribeStatus(listener: StatusListener): Unsubscribe;
    ensureReady(options?: ReadyOptions): Promise<void>;
    detectTextRegions(image: ImagePixels, options?: ReadyOptions): Promise<readonly TextRegion[]>;
}
