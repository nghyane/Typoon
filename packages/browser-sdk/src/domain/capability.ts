export type CapabilityState = 'idle' | 'resolving' | 'downloading' | 'initializing' | 'ready' | 'failed'

export interface CapabilityProgress {
  readonly receivedBytes?: number
  readonly totalBytes?: number
  readonly ratio?: number
}

export interface CapabilityStatus {
  readonly name: string
  readonly state: CapabilityState
  readonly progress?: CapabilityProgress
  readonly error?: unknown
}

export interface StatusSnapshot {
  readonly capabilities: readonly CapabilityStatus[]
}

export interface ReadyOptions {
  readonly signal?: AbortSignal
}

export type StatusListener<T = CapabilityStatus> = (status: T) => void
export type Unsubscribe = () => void

export interface Capability {
  status(): CapabilityStatus
  subscribeStatus(listener: StatusListener): Unsubscribe
  ensureReady(options?: ReadyOptions): Promise<void>
}

export function isCapability(value: unknown): value is Capability {
  return !!value
    && typeof value === 'object'
    && typeof (value as Partial<Capability>).status === 'function'
    && typeof (value as Partial<Capability>).subscribeStatus === 'function'
    && typeof (value as Partial<Capability>).ensureReady === 'function'
}
