import type { ModelLoader } from '../../models/ModelLoader';
import { type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../../domain/capability';
import type { ImagePixels } from '../../domain/image';
import type { TextRegion } from '../../domain/regions';
import type { TextRegionRunner } from './TextRegionRunner';
import type { ComicDetrProvider } from './ortTypes';
import type { OrtSessionPool } from '../../models/OrtSessionPool';
export interface MainThreadOrtRunnerOptions {
    readonly model: ModelLoader;
    readonly confidenceThreshold: number;
    readonly providers?: readonly ComicDetrProvider[];
    readonly sessionPool: OrtSessionPool;
}
export declare class MainThreadOrtRunner implements TextRegionRunner {
    readonly name = "manga-text-region-runner";
    private readonly capability;
    private readonly options;
    private sessionPromise;
    constructor(options: MainThreadOrtRunnerOptions);
    status(): CapabilityStatus;
    subscribeStatus(listener: StatusListener): Unsubscribe;
    ensureReady(options?: ReadyOptions): Promise<void>;
    run(image: ImagePixels, options?: ReadyOptions): Promise<readonly TextRegion[]>;
    private getSession;
    private createSession;
}
