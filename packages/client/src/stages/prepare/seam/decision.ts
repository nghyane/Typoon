import type { SeamDecisionPolicy } from '../../../domain/prepare'
import type { SeamSignals } from './signals'

export interface SeamDecision {
  readonly action: 'merge' | 'cut' | 'uncertain'
  readonly confidence: number
  readonly reason: string
}

export function decideSeam(signals: SeamSignals, policy: SeamDecisionPolicy): SeamDecision {
  const mergeScore = weightedMergeScore(signals)
  const cutScore = weightedCutScore(signals)
  const crossingScore = Math.max(
    signals.bubbleComponentCrossing.score,
    signals.textInkCrossing.score,
    signals.edgeContinuity.score,
  )

  if (crossingScore >= 0.42 && mergeScore >= 0.45) {
    return { action: 'merge', confidence: Math.max(mergeScore, crossingScore), reason: 'content-crosses-seam' }
  }

  if (mergeScore >= policy.mergeConfidence && mergeScore > cutScore) {
    return { action: 'merge', confidence: mergeScore, reason: 'semantic-component-crosses-seam' }
  }
  if (cutScore >= policy.cutConfidence && cutScore > mergeScore) {
    return { action: 'cut', confidence: cutScore, reason: 'clean-boundary' }
  }
  return { action: 'uncertain', confidence: Math.max(mergeScore, cutScore), reason: 'ambiguous-seam' }
}

function weightedMergeScore(signals: SeamSignals): number {
  return clamp01(
    signals.bubbleComponentCrossing.score * 0.50 +
    signals.textInkCrossing.score * 0.30 +
    signals.edgeContinuity.score * 0.15 +
    (1 - signals.panelGutter.score) * 0.05,
  )
}

function weightedCutScore(signals: SeamSignals): number {
  return clamp01(
    signals.panelGutter.score * 0.55 +
    (1 - signals.bubbleComponentCrossing.score) * 0.25 +
    (1 - signals.textInkCrossing.score) * 0.15 +
    (1 - signals.edgeContinuity.score) * 0.05,
  )
}

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n))
}
