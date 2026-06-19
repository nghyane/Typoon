import type { CapabilityProgress, CapabilityStatus, StatusListener, Unsubscribe } from '../domain/capability';
export declare class CapabilityMachine {
    private current;
    private readonly name;
    private readonly listeners;
    constructor(name: string);
    status(): CapabilityStatus;
    subscribe(listener: StatusListener): Unsubscribe;
    resolving(): void;
    downloading(progress: CapabilityProgress): void;
    initializing(): void;
    ready(): void;
    failed(error: unknown): void;
    mirror(status: CapabilityStatus): void;
    private publish;
}
