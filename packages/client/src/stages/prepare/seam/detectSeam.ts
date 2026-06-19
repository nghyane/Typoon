import type { PrepareProfile } from '../../../domain/prepare'
import type { ImagePixels } from '../../../domain/image'
import type { CanvasBackend } from '../canvasBackend'
import { analyzeSeamSignals, type SeamSignals } from './signals'
import { decideSeam, type SeamDecision } from './decision'

export interface SeamAnalysis {
  readonly boundary: {
    readonly topSourcePageIndex: number
    readonly bottomSourcePageIndex: number
  }
  readonly decision: SeamDecision
  readonly signals: SeamSignals
  readonly bandPx: number
}

export async function detectSeam(args: {
  readonly topSourcePageIndex: number
  readonly bottomSourcePageIndex: number
  readonly top: ImagePixels
  readonly bottom: ImagePixels
  readonly profile: PrepareProfile
  readonly backend: CanvasBackend
  readonly signal?: AbortSignal
}): Promise<SeamAnalysis> {
  const bandPx = seamBandPx(args.top, args.bottom, args.profile)
  const topBand = await args.backend.downsampleBand({
    image: args.top,
    edge: 'bottom',
    bandPx,
    targetWidthPx: args.profile.seam.previewWidthPx,
    signal: args.signal,
  })
  const bottomBand = await args.backend.downsampleBand({
    image: args.bottom,
    edge: 'top',
    bandPx,
    targetWidthPx: args.profile.seam.previewWidthPx,
    signal: args.signal,
  })
  const signals = analyzeSeamSignals({ topBand, bottomBand })
  return {
    boundary: {
      topSourcePageIndex: args.topSourcePageIndex,
      bottomSourcePageIndex: args.bottomSourcePageIndex,
    },
    decision: decideSeam(signals, args.profile.seam.decision),
    signals,
    bandPx,
  }
}

function seamBandPx(top: ImagePixels, bottom: ImagePixels, profile: PrepareProfile): number {
  const base = Math.round(Math.min(top.height, bottom.height) * profile.seam.bandFraction)
  return Math.max(profile.seam.minBandPx, Math.min(profile.seam.maxBandPx, base))
}
