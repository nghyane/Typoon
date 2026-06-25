// pipeline/bubbleRecovery.ts — Phase B authoritative OCR per DETR region.
//
// Single-pass Lens on the whole capture under-samples small bubbles, dropping
// their text entirely or clipping edge glyphs. This stage walks the Comic-DETR
// anchors, finds the ones the coarse pass left EMPTY or PARTIAL, and re-OCRs a
// tight per-bubble crop **from the full-resolution source page** (upscaled so
// Lens has enough glyph resolution). The recovered blocks replace the
// incomplete members; COMPLETE anchors are left untouched.
// Ported from the legacy Python `lens/bubble_pass.py`.

import type { BBox } from '../domain/geometry'
import type { ImagePixels } from '../domain/image'
import type { RecognizedTextPage, TextBlock } from '../domain/text'
import type { TextRegion } from '../domain/regions'
import { AsyncLimiter } from '../flow/AsyncLimiter'

// One re-OCR per spatial cluster of DETR regions. text_bubble is the tightest
// inner rect so it wins; bubble next; text_free for captions outside balloons.
const ANCHOR_PRECEDENCE: readonly TextRegion['kind'][] = ['text_bubble', 'bubble', 'text_free']
const CLUSTER_IOU = 0.5
// A side gap larger than this fraction of the median line height means the
// coarse pass missed a chunk of the bubble (edge-glyph drop / cut line).
const GAP_THRESHOLD_FACTOR = 0.7
const CROP_PAD_SOURCE_PX = 6
// Lens recognition collapses below ~200 px short side; upscale crops up to here.
const MIN_CROP_DIM = 200
// Bound concurrent per-bubble Lens calls so a dense page does not flood the proxy.
const MAX_CONCURRENT_RECOVERY = 6

export interface BubbleCropRecognizer {
  recognizeCrop(image: ImagePixels): Promise<RecognizedTextPage>
}

/** Coordinate mapping + source-image access for Phase B crops. */
export interface BubbleSource {
  /** Load the full-resolution stitched canvas (page N + both halos) at 1:1,
   *  shared across all anchor crops for this page.  Origin matches the capture
   *  canvas, so capture-space bbox → source-space = bbox / captureScale. */
  readonly loadFullCanvas: () => Promise<HTMLCanvasElement>
  /** Scale factor: source px → capture px. */
  readonly captureScale: number
}

type Action = 'complete' | 'empty' | 'partial'

interface RecoveryAnchor {
  readonly kind: TextRegion['kind']
  readonly bbox: BBox
}

interface Diagnosis {
  readonly anchor: RecoveryAnchor
  readonly action: Action
  readonly memberIndices: readonly number[]
}

/**
 * Re-OCR incomplete bubbles and splice the recovered text into `recognized`.
 * Returns the original page unchanged when there are no regions/anchors or
 * nothing needs recovery.
 */
export async function recoverBubbleText(args: {
  readonly recognized: RecognizedTextPage
  readonly source: BubbleSource
  readonly regions: readonly TextRegion[]
  readonly recognizer: BubbleCropRecognizer
}): Promise<RecognizedTextPage> {
  const { recognized, source, regions, recognizer } = args
  if (!regions.length) return recognized

  const anchors = selectAnchors(regions)
  if (!anchors.length) return recognized

  const blocks = recognized.blocks
  const diagnoses = anchors.map(anchor => diagnose(anchor, blocks))
  const todo = diagnoses.filter(d => d.action !== 'complete')
  if (!todo.length) return recognized

  // Load full-res source once; all anchor crops share the same canvas.
  const fullCanvas = await source.loadFullCanvas()

  const limiter = new AsyncLimiter(MAX_CONCURRENT_RECOVERY)
  const recovered = await Promise.all(
    todo.map(d => limiter.run(() => ocrAnchor(recognizer, fullCanvas, source, d.anchor))),
  )
  const merged = splice(blocks, todo, recovered)
  return { ...recognized, blocks: merged }
}

// ── Anchor selection: one anchor per spatial cluster ──────────────────────

function selectAnchors(regions: readonly TextRegion[]): RecoveryAnchor[] {
  const candidates = regions.filter(region => ANCHOR_PRECEDENCE.includes(region.kind))
  const out: RecoveryAnchor[] = []
  for (const cluster of clusterRegions(candidates, CLUSTER_IOU)) {
    const winner = pickClusterAnchor(cluster)
    if (winner) out.push(winner)
  }
  return out
}

function clusterRegions(regions: readonly TextRegion[], iouThreshold: number): TextRegion[][] {
  const parent = regions.map((_, i) => i)
  const find = (i: number): number => {
    let root = i
    while (parent[root] !== root) root = parent[root]!
    let cur = i
    while (parent[cur] !== root) {
      const next = parent[cur]!
      parent[cur] = root
      cur = next
    }
    return root
  }
  const union = (i: number, j: number): void => {
    const ri = find(i)
    const rj = find(j)
    if (ri !== rj) parent[ri] = rj
  }
  for (let i = 0; i < regions.length; i += 1) {
    for (let j = i + 1; j < regions.length; j += 1) {
      if (iou(regions[i]!.bbox, regions[j]!.bbox) > iouThreshold) union(i, j)
    }
  }
  const buckets = new Map<number, TextRegion[]>()
  for (let i = 0; i < regions.length; i += 1) {
    const root = find(i)
    const bucket = buckets.get(root)
    if (bucket) bucket.push(regions[i]!)
    else buckets.set(root, [regions[i]!])
  }
  return [...buckets.values()]
}

function pickClusterAnchor(cluster: readonly TextRegion[]): RecoveryAnchor | null {
  for (const kind of ANCHOR_PRECEDENCE) {
    const same = cluster.filter(region => region.kind === kind)
    if (same.length) {
      const best = same.reduce((a, b) => (b.confidence > a.confidence ? b : a))
      return { kind: best.kind, bbox: best.bbox }
    }
  }
  return null
}

// ── Diagnosis: COMPLETE / EMPTY / PARTIAL ─────────────────────────────────

function diagnose(anchor: RecoveryAnchor, blocks: readonly TextBlock[]): Diagnosis {
  const memberIndices = blocks
    .map((block, index) => ({ block, index }))
    .filter(({ block }) => centerInside(block.bbox, anchor.bbox))
    .map(({ index }) => index)
  if (!memberIndices.length) return { anchor, action: 'empty', memberIndices: [] }

  const members = memberIndices.map(i => blocks[i]!)
  const lineHeight = medianLineHeight(members)
  if (lineHeight === 0) return { anchor, action: 'complete', memberIndices }

  const wu = wordUnion(members)
  const gapTop = wu[1] - anchor.bbox[1]
  const gapBottom = anchor.bbox[3] - wu[3]
  const gapLeft = wu[0] - anchor.bbox[0]
  const gapRight = anchor.bbox[2] - wu[2]
  const threshold = GAP_THRESHOLD_FACTOR * lineHeight
  if (Math.max(gapTop, gapBottom, gapLeft, gapRight) > threshold) {
    return { anchor, action: 'partial', memberIndices }
  }
  return { anchor, action: 'complete', memberIndices }
}

// ── Re-OCR a single anchor crop from full-resolution source ────────────────

async function ocrAnchor(
  recognizer: BubbleCropRecognizer,
  fullCanvas: HTMLCanvasElement,
  source: BubbleSource,
  anchor: RecoveryAnchor,
): Promise<TextBlock[]> {
  const { captureScale } = source
  const sw = fullCanvas.width
  const sh = fullCanvas.height

  // Convert capture-space bbox to stitched-canvas space.
  // Stitched canvas = capture canvas at full resolution (page N + both halos).
  // Origin matches → sx = cx / captureScale, sy = cy / captureScale.
  const sx1 = anchor.bbox[0] / captureScale
  const sy1 = anchor.bbox[1] / captureScale
  const sx2 = anchor.bbox[2] / captureScale
  const sy2 = anchor.bbox[3] / captureScale

  const pad = Math.round(CROP_PAD_SOURCE_PX / captureScale)
  const csx1 = Math.max(0, Math.floor(sx1 - pad))
  const csy1 = Math.max(0, Math.floor(sy1 - pad))
  const csx2 = Math.min(sw, Math.ceil(sx2 + pad))
  const csy2 = Math.min(sh, Math.ceil(sy2 + pad))
  const cw = csx2 - csx1
  const ch = csy2 - csy1
  if (cw <= 0 || ch <= 0) return []

  const upscale = Math.max(1, Math.ceil(MIN_CROP_DIM / Math.max(1, Math.min(cw, ch))))
  const crop = cropCanvasRegion(fullCanvas, csx1, csy1, cw, ch, upscale)
  let recognized: RecognizedTextPage
  try {
    recognized = await recognizer.recognizeCrop(crop)
  } catch {
    return []
  }

  // Blocks are in upscaled crop pixels. Convert: unscale → add crop origin (source
  // space) → scale back to capture space.
  return recognized.blocks.map(block =>
    transformBlock(block, csx1, csy1, upscale, captureScale),
  )
}

function cropCanvasRegion(
  source: HTMLCanvasElement,
  x: number,
  y: number,
  w: number,
  h: number,
  scale: number,
): ImagePixels {
  const out = document.createElement('canvas')
  out.width = Math.max(1, Math.round(w * scale))
  out.height = Math.max(1, Math.round(h * scale))
  const outCtx = out.getContext('2d')
  if (!outCtx) throw new Error('2d canvas unavailable')
  outCtx.imageSmoothingEnabled = true
  outCtx.imageSmoothingQuality = 'high'
  outCtx.drawImage(source, x, y, w, h, 0, 0, out.width, out.height)
  const pixels = outCtx.getImageData(0, 0, out.width, out.height)
  return { width: out.width, height: out.height, data: pixels.data }
}

// ── Splice: drop replaced members, append recovered ──────────────────────

function splice(
  blocks: readonly TextBlock[],
  diagnoses: readonly Diagnosis[],
  recovered: readonly TextBlock[][],
): TextBlock[] {
  const drop = new Set<number>()
  for (let i = 0; i < diagnoses.length; i += 1) {
    // Lens returned nothing — keep the originals as-is.
    if (recovered[i]!.length) for (const index of diagnoses[i]!.memberIndices) drop.add(index)
  }
  const kept = blocks.filter((_, index) => !drop.has(index))
  return [...kept, ...recovered.flat()]
}

// ── Coordinate transform: crop → source → capture ─────────────────────────

function transformBlock(
  block: TextBlock,
  cropSrcX: number,
  cropSrcY: number,
  upscale: number,
  captureScale: number,
): TextBlock {
  const t = (b: BBox): BBox => toCapture(b, cropSrcX, cropSrcY, upscale, captureScale)
  return {
    ...block,
    bbox: t(block.bbox),
    polygon: block.polygon.map(([px, py]) => {
      const pt = t([px, py, px, py])
      return [pt[0], pt[1]]
    }),
    lines: block.lines.map(line => ({
      ...line,
      bbox: t(line.bbox),
      words: line.words.map(word => ({ ...word, bbox: t(word.bbox) })),
    })),
    words: block.words.map(word => ({ ...word, bbox: t(word.bbox) })),
  }
}

function toCapture(
  bbox: BBox,
  cropSrcX: number,
  cropSrcY: number,
  upscale: number,
  captureScale: number,
): BBox {
  const sx = (cropSrcX + bbox[0] / upscale) * captureScale
  const sy = (cropSrcY + bbox[1] / upscale) * captureScale
  const ex = (cropSrcX + bbox[2] / upscale) * captureScale
  const ey = (cropSrcY + bbox[3] / upscale) * captureScale
  return [sx, sy, ex, ey]
}

// ── Geometry helpers ──────────────────────────────────────────────────────

function centerInside(inner: BBox, outer: BBox): boolean {
  const cx = (inner[0] + inner[2]) / 2
  const cy = (inner[1] + inner[3]) / 2
  return outer[0] <= cx && cx <= outer[2] && outer[1] <= cy && cy <= outer[3]
}

function wordUnion(members: readonly TextBlock[]): BBox {
  const boxes = members.flatMap(m => (m.words.length ? m.words.map(w => w.bbox) : [m.bbox]))
  return [
    Math.min(...boxes.map(b => b[0])),
    Math.min(...boxes.map(b => b[1])),
    Math.max(...boxes.map(b => b[2])),
    Math.max(...boxes.map(b => b[3])),
  ]
}

function medianLineHeight(members: readonly TextBlock[]): number {
  const heights: number[] = []
  for (const m of members) {
    for (const line of m.lines) {
      heights.push(Math.min(Math.max(1, line.bbox[2] - line.bbox[0]), Math.max(1, line.bbox[3] - line.bbox[1])))
    }
  }
  if (!heights.length) return 0
  heights.sort((a, b) => a - b)
  return heights[Math.floor(heights.length / 2)] ?? 0
}

function iou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  const aa = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]))
  const bb = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]))
  return inter / (aa + bb - inter)
}
