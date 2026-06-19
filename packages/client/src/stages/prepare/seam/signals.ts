import type { ImagePixels } from '../../../domain/image'

export interface SignalScore {
  readonly score: number
  readonly confidence: number
}

export interface SeamSignals {
  readonly bubbleComponentCrossing: SignalScore
  readonly textInkCrossing: SignalScore
  readonly panelGutter: SignalScore
  readonly edgeContinuity: SignalScore
}

export function analyzeSeamSignals(args: {
  readonly topBand: ImagePixels
  readonly bottomBand: ImagePixels
}): SeamSignals {
  const width = Math.min(args.topBand.width, args.bottomBand.width)
  const bubble = whiteBridge(args.topBand, args.bottomBand, width)
  const ink = darkBridge(args.topBand, args.bottomBand, width)
  const gutter = cleanGutter(args.topBand, args.bottomBand, width)
  const edge = edgeBridge(args.topBand, args.bottomBand, width)
  return {
    bubbleComponentCrossing: { score: bubble, confidence: confidenceFromSamples(width, 3) },
    textInkCrossing: { score: ink, confidence: confidenceFromSamples(width, 3) },
    panelGutter: { score: gutter, confidence: confidenceFromSamples(width, 4) },
    edgeContinuity: { score: edge, confidence: confidenceFromSamples(width, 3) },
  }
}

function whiteBridge(top: ImagePixels, bottom: ImagePixels, width: number): number {
  let columns = 0
  let bridged = 0
  for (let x = 0; x < width; x += 3) {
    columns += 1
    if (edgeWhiteRun(top, x, 'bottom') > top.height * 0.35 && edgeWhiteRun(bottom, x, 'top') > bottom.height * 0.35) bridged += 1
  }
  return columns ? bridged / columns : 0
}

function darkBridge(top: ImagePixels, bottom: ImagePixels, width: number): number {
  let columns = 0
  let bridged = 0
  for (let x = 0; x < width; x += 3) {
    columns += 1
    const topDark = hasDarkPixel(top, x, Math.floor(top.height * 0.55), top.height)
    const bottomDark = hasDarkPixel(bottom, x, 0, Math.ceil(bottom.height * 0.45))
    if (topDark && bottomDark) bridged += 1
  }
  return columns ? bridged / columns : 0
}

function cleanGutter(top: ImagePixels, bottom: ImagePixels, width: number): number {
  let columns = 0
  let blankColumns = 0
  for (let x = 0; x < width; x += 4) {
    columns += 1
    if (isBlankColumn(top, x, Math.floor(top.height * 0.82), top.height)
      && isBlankColumn(bottom, x, 0, Math.ceil(bottom.height * 0.18))) {
      blankColumns += 1
    }
  }
  const blankRatio = columns ? blankColumns / columns : 0
  if (blankRatio < 0.82) return blankRatio * 0.5
  return blankRatio
}

function isBlankColumn(image: ImagePixels, x: number, y1: number, y2: number): boolean {
  let samples = 0
  let blank = 0
  for (let y = y1; y < y2; y += 3) {
    samples += 1
    const luma = pixelLuma(image, x, y)
    if (luma > 245) blank += 1
  }
  return samples > 0 && blank / samples > 0.92
}

function edgeBridge(top: ImagePixels, bottom: ImagePixels, width: number): number {
  let columns = 0
  let bridged = 0
  for (let x = 1; x < width - 1; x += 3) {
    columns += 1
    const topEdge = maxVerticalGradient(top, x, Math.floor(top.height * 0.55), top.height)
    const bottomEdge = maxVerticalGradient(bottom, x, 0, Math.ceil(bottom.height * 0.45))
    if (topEdge > 80 && bottomEdge > 80) bridged += 1
  }
  return columns ? bridged / columns : 0
}

function edgeWhiteRun(image: ImagePixels, x: number, edge: 'top' | 'bottom'): number {
  let run = 0
  if (edge === 'top') {
    for (let y = 0; y < image.height; y += 2) {
      if (pixelLuma(image, x, y) > 235) run += 2
      else break
    }
    return run
  }
  for (let y = image.height - 1; y >= 0; y -= 2) {
    if (pixelLuma(image, x, y) > 235) run += 2
    else break
  }
  return run
}

function hasDarkPixel(image: ImagePixels, x: number, y1: number, y2: number): boolean {
  for (let y = y1; y < y2; y += 2) {
    if (pixelLuma(image, x, y) < 80) return true
  }
  return false
}

function maxVerticalGradient(image: ImagePixels, x: number, y1: number, y2: number): number {
  let max = 0
  for (let y = Math.max(1, y1); y < Math.min(image.height - 1, y2); y += 2) {
    max = Math.max(max, Math.abs(pixelLuma(image, x, y - 1) - pixelLuma(image, x, y + 1)))
  }
  return max
}

function pixelLuma(image: ImagePixels, x: number, y: number): number {
  const xx = Math.max(0, Math.min(image.width - 1, Math.floor(x)))
  const yy = Math.max(0, Math.min(image.height - 1, Math.floor(y)))
  const i = (yy * image.width + xx) * 4
  const r = image.data[i] ?? 255
  const g = image.data[i + 1] ?? 255
  const b = image.data[i + 2] ?? 255
  return 0.299 * r + 0.587 * g + 0.114 * b
}

function confidenceFromSamples(width: number, step: number): number {
  return Math.min(1, Math.max(0.2, Math.floor(width / step) / 160))
}
