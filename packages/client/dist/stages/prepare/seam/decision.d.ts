import type { SeamDecisionPolicy } from '../../../domain/prepare';
import type { SeamSignals } from './signals';
export interface SeamDecision {
    readonly action: 'merge' | 'cut' | 'uncertain';
    readonly confidence: number;
    readonly reason: string;
}
export declare function decideSeam(signals: SeamSignals, policy: SeamDecisionPolicy): SeamDecision;
