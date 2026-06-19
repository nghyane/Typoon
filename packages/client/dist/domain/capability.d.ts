export type CapabilityState = 'idle' | 'resolving' | 'downloading' | 'initializing' | 'ready' | 'failed';
export interface CapabilityProgress {
    readonly receivedBytes: number;
    readonly totalBytes?: number;
    readonly ratio?: number;
}
interface CapabilityStatusBase {
    readonly name: string;
}
export type CapabilityStatus = (CapabilityStatusBase & {
    readonly state: 'idle';
}) | (CapabilityStatusBase & {
    readonly state: 'resolving';
}) | (CapabilityStatusBase & {
    readonly state: 'downloading';
    readonly progress: CapabilityProgress;
}) | (CapabilityStatusBase & {
    readonly state: 'initializing';
}) | (CapabilityStatusBase & {
    readonly state: 'ready';
}) | (CapabilityStatusBase & {
    readonly state: 'failed';
    readonly error: unknown;
});
export interface StatusSnapshot {
    readonly capabilities: readonly CapabilityStatus[];
}
export interface ReadyOptions {
    readonly signal?: AbortSignal;
}
export type StatusListener<T = CapabilityStatus> = (status: T) => void;
export type Unsubscribe = () => void;
export interface Capability {
    status(): CapabilityStatus;
    subscribeStatus(listener: StatusListener): Unsubscribe;
    ensureReady(options?: ReadyOptions): Promise<void>;
}
export declare function isCapability(value: unknown): value is Capability;
export {};
