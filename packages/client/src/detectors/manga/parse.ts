import type * as ort from 'onnxruntime-web/wasm'
import type { TextRegion } from '../../domain/regions'

const BBOX_PAD_PX = 6
const CLASS_NAMES = ['bubble', 'text_bubble', 'text_free'] as const
const NUM_CLASSES = CLASS_NAMES.length

/**
 * Decode Comic-DETR's raw head outputs into text regions.
 *
 * The model is exported WITHOUT its postprocessor: the in-graph TopK/Gather and
 * the int64 `orig_target_sizes` box scaling are removed because WebGPU can't run
 * those ops, so they forced GPU→CPU→GPU copies every page. The model now emits
 * two per-query tensors and we do the (cheap) decode here:
 *   scores [1, Q, C]  already sigmoid-activated, row-major  → index q*C + c
 *   boxes  [1, Q, 4]  xyxy in normalised [0,1] coords        → index q*4 + k
 * Thresholding every (query, class) pair and scaling to page pixels is
 * equivalent to the original postprocessor's topk-then-threshold for any
 * realistic detection count (< Q).
 */
export function parseDetections(
  output: Record<string, ort.Tensor>,
  outputNames: readonly string[],
  pageW: number,
  pageH: number,
  confidenceThreshold: number,
): TextRegion[] {
  const scores = tensorByName(output, outputNames, 'scores')
  const boxes = tensorByName(output, outputNames, 'boxes')
  const scoreData = scores.data as Float32Array
  const boxData = boxes.data as Float32Array
  const queries = Math.floor(scoreData.length / NUM_CLASSES)
  const out: TextRegion[] = []
  for (let q = 0; q < queries; q++) {
    const offset = q * 4
    for (let c = 0; c < NUM_CLASSES; c++) {
      const score = scoreData[q * NUM_CLASSES + c] ?? 0
      if (score < confidenceThreshold) continue
      const className = CLASS_NAMES[c]
      if (!className) continue
      const x1 = clamp(Math.floor((boxData[offset] ?? 0) * pageW) - BBOX_PAD_PX, 0, pageW)
      const y1 = clamp(Math.floor((boxData[offset + 1] ?? 0) * pageH) - BBOX_PAD_PX, 0, pageH)
      const x2 = clamp(Math.ceil((boxData[offset + 2] ?? 0) * pageW) + BBOX_PAD_PX, 0, pageW)
      const y2 = clamp(Math.ceil((boxData[offset + 3] ?? 0) * pageH) + BBOX_PAD_PX, 0, pageH)
      if (x2 <= x1 || y2 <= y1) continue
      out.push({ kind: className, bbox: [x1, y1, x2, y2], confidence: score })
    }
  }
  return out
}

function tensorByName(output: Record<string, ort.Tensor>, outputNames: readonly string[], suffix: string): ort.Tensor {
  const name = outputNames.find(candidate => candidate.toLowerCase().includes(suffix)) ?? suffix
  const tensor = output[name]
  if (!tensor) throw new Error(`missing comic-detr output tensor: ${suffix}`)
  return tensor
}

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n))
}
