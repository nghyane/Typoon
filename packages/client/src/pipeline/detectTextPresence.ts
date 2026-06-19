import type { ImagePixels } from '../domain/image'

/**
 * Lightweight text-presence detector — no ML model, runs in < 1 ms.
 * Pixels stay in the ArrayBuffer, no canvas/GPU round-trip.
 *
 * Algorithm: grid-sample the page into N×N blocks.  For each block compute
 * the luminance standard deviation (σ).  Text produces high local variance;
 * blank / gradient-fill produces low variance.  If enough blocks exceed the
 * noise floor the page has text.
 *
 * This is more robust than edge-detection because it captures the texture
 * density of text (many small high-contrast transitions in a local area)
 * rather than individual pixel-to-pixel differences.
 */
const GRID_SIZE = 12
const BLOCK_SAMPLE_STEP = 3
const VARIANCE_TEXT_THRESHOLD = 180
const MIN_TEXT_BLOCKS = 1

export interface TextPresenceResult {
  readonly hasText: boolean
  /** Fraction of grid blocks that contain text-like texture. */
  readonly textBlockFraction: number
}

export function detectTextPresence(image: ImagePixels): TextPresenceResult {
  const blockW = Math.max(1, Math.floor(image.width / GRID_SIZE))
  const blockH = Math.max(1, Math.floor(image.height / GRID_SIZE))
  if (blockW < 4 || blockH < 4) return { hasText: true, textBlockFraction: 1 }

  let textBlocks = 0
  let totalBlocks = 0

  for (let by = 0; by < GRID_SIZE; by++) {
    const y1 = by * blockH
    const y2 = Math.min(image.height, y1 + blockH)

    for (let bx = 0; bx < GRID_SIZE; bx++) {
      const x1 = bx * blockW
      const x2 = Math.min(image.width, x1 + blockW)
      totalBlocks++

      const variance = blockLumaVariance(image, x1, y1, x2, y2)
      if (variance > VARIANCE_TEXT_THRESHOLD) textBlocks++
    }
  }

  const fraction = textBlocks / totalBlocks
  return { hasText: textBlocks >= MIN_TEXT_BLOCKS, textBlockFraction: fraction }
}

function blockLumaVariance(image: ImagePixels, x1: number, y1: number, x2: number, y2: number): number {
  let sum = 0
  let sumSq = 0
  let count = 0

  for (let y = y1; y < y2; y += BLOCK_SAMPLE_STEP) {
    for (let x = x1; x < x2; x += BLOCK_SAMPLE_STEP) {
      const luma = lumaAt(image, x, y)
      sum += luma
      sumSq += luma * luma
      count++
    }
  }

  if (count < 4) return 0
  const mean = sum / count
  return sumSq / count - mean * mean
}

function lumaAt(image: ImagePixels, x: number, y: number): number {
  const xx = Math.max(0, Math.min(image.width - 1, Math.floor(x)))
  const yy = Math.max(0, Math.min(image.height - 1, Math.floor(y)))
  const i = (yy * image.width + xx) * 4
  const r = image.data[i] ?? 255
  const g = image.data[i + 1] ?? 255
  const b = image.data[i + 2] ?? 255
  return 0.299 * r + 0.587 * g + 0.114 * b
}
