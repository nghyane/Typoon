import { type Capability, type CapabilityStatus, type ReadyOptions, type StatusListener, type Unsubscribe } from '../domain/capability';
import type { ModelDescriptor } from '../domain/model';
import type { ModelRegistry } from './ModelRegistry';
import type { ModelStore } from './ModelStore';
export declare class ModelLoader implements Capability {
    readonly id: string;
    readonly name: string;
    readonly descriptor: ModelDescriptor;
    private readonly capability;
    private readonly store;
    private bytesPromise;
    constructor(id: string, registry: ModelRegistry, store: ModelStore);
    status(): CapabilityStatus;
    subscribeStatus(listener: StatusListener): Unsubscribe;
    ensureReady(options?: ReadyOptions): Promise<void>;
    bytes(options?: ReadyOptions): Promise<ArrayBuffer>;
    private load;
}
