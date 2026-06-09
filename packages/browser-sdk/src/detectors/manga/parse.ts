import * as ort from 'onnxruntime-web/webgpu'
import type { TextRegion } from '../../domain/regions'

const BBOX_PAD_PX = 6
const CLASS_NAMES = ['bubble', 'text_bubble', 'text_free'] as const

export function parseDetections(
  output: Record<string, ort.Tensor>,
  outputNames: readonly string[],
  pageW: number,
  pageH: number,
  confidenceThreshold: number,
): TextRegion[] {
  const labels = tensorByName(output, outputNames, 'labels')
  const boxes = tensorByName(output, outputNames, 'boxes')
  const scores = tensorByName(output, outputNames, 'scores')
  const labelData = labels.data as BigInt64Array | Int32Array | Float32Array
  const boxData = boxes.data as Float32Array
  const scoreData = scores.data as Float32Array
  const out: TextRegion[] = []
  for (let i = 0; i < scoreData.length; i++) {
    const score = Number(scoreData[i] ?? 0)
    if (score < confidenceThreshold) continue
    const clsId = Number(labelData[i] ?? -1)
    const className = CLASS_NAMES[clsId]
    if (!className) continue
    const offset = i * 4
    const x1 = clamp(Math.floor(Number(boxData[offset] ?? 0)) - BBOX_PAD_PX, 0, pageW)
    const y1 = clamp(Math.floor(Number(boxData[offset + 1] ?? 0)) - BBOX_PAD_PX, 0, pageH)
    const x2 = clamp(Math.ceil(Number(boxData[offset + 2] ?? 0)) + BBOX_PAD_PX, 0, pageW)
    const y2 = clamp(Math.ceil(Number(boxData[offset + 3] ?? 0)) + BBOX_PAD_PX, 0, pageH)
    if (x2 <= x1 || y2 <= y1) continue
    out.push({ kind: className, bbox: [x1, y1, x2, y2], confidence: score })
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
