import type { CapabilityProgress } from '../domain/capability';
import type { ModelDescriptor } from './modelTypes';
export declare function loadModelBytes(descriptor: ModelDescriptor, onProgress: (progress: CapabilityProgress) => void, signal: AbortSignal | undefined): Promise<ArrayBuffer>;
