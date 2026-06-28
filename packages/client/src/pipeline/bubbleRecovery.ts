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
import { isArtifactBlock } from './ocrArtifacts'

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
// A pre-existing block this much inside a recovered anchor is superseded by the
// authoritative crop re-OCR and must be dropped — otherwise a coarse block that
// overlaps the balloon but whose *centre* fell outside the anchor (so it was not
// a diagnosed member) survives next to the recovered text and renders as a
// duplicate (e.g. a top line OCR'd separately).
const ANCHOR_REPLACE_CONTAINMENT = 0.5
// Within one balloon's re-OCR, a smaller block this much inside a larger sibling
// is a spurious fragment (upscaled crops let Lens read faint edge/strokes as a
// stray rotated line). One balloon ⇒ one coherent text block; drop the rest.
const RECOVERED_SUBBLOCK_CONTAINMENT = 0.5

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

  // Concurrency is bounded globally inside the recognizer (one semaphore across
  // all pages), so crops can fan out freely here.
  const recovered = await Promise.all(
    todo.map(d => ocrAnchor(recognizer, fullCanvas, source, d.anchor)),
  )
  const merged = splice(blocks, todo, recovered)
  return { ...recognized, blocks: merged }
}

// ── In-bubble ghost hygiene (runs for complete AND recovered balloons) ─────
// One balloon holds one coherent text cluster. A block whose centre sits inside
// a DETR balloon but which is mostly contained inside a richer sibling there is
// a ghost — a duplicate copy, or a stray (often rotated) fragment the OCR
// invented over the real dialogue. Drop it before grouping so it never becomes
// its own placement rendered on top of the bubble. Stacked real lines do not
// contain one another, so genuine text is untouched.

const GHOST_CONTAINMENT = 0.5

export function removeInBubbleGhostBlocks(
  recognized: RecognizedTextPage,
  regions: readonly TextRegion[],
): RecognizedTextPage {
  const balloons = regions.filter(region => region.kind === 'text_bubble' || region.kind === 'bubble')
  if (!balloons.length) return recognized

  const blocks = recognized.blocks
  const drop = new Set<number>()
  for (const balloon of balloons) {
    const members = blocks
      .map((block, index) => ({ block, index }))
      .filter(({ block, index }) => !drop.has(index) && centerInside(block.bbox, balloon.bbox))
    if (members.length < 2) continue
    // Richest text first, then larger area, so the real dialogue is the survivor.
    members.sort((a, b) => textWeight(b.block) - textWeight(a.block) || bboxArea(b.block.bbox) - bboxArea(a.block.bbox))
    const kept: TextBlock[] = []
    for (const { block, index } of members) {
      if (kept.some(k => containment(block.bbox, k.bbox) >= GHOST_CONTAINMENT)) drop.add(index)
      else kept.push(block)
    }
  }
  if (!drop.size) return recognized
  return { ...recognized, blocks: blocks.filter((_, index) => !drop.has(index)) }
}

function textWeight(block: TextBlock): number {
  return [...block.text].filter(ch => !/\s/u.test(ch)).length
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
      if (sameCluster(regions[i]!.bbox, regions[j]!.bbox, iouThreshold)) union(i, j)
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

// A DETR `bubble` and the `text_bubble`/`text_free` nested inside it describe
// the same balloon, but their IoU is low because the inner box is much smaller,
// so a pure-IoU link leaves them as two anchors and re-OCRs the bubble twice —
// producing duplicate blocks that grouping then concatenates into one doubled
// placement. Link by containment of the smaller box too: one anchor per balloon.
const CLUSTER_CONTAINMENT = 0.7

function sameCluster(a: BBox, b: BBox, iouThreshold: number): boolean {
  if (iou(a, b) > iouThreshold) return true
  const [inner, outer] = bboxArea(a) <= bboxArea(b) ? [a, b] : [b, a]
  return containment(inner, outer) >= CLUSTER_CONTAINMENT
}

function containment(inner: BBox, outer: BBox): number {
  const ix1 = Math.max(inner[0], outer[0])
  const iy1 = Math.max(inner[1], outer[1])
  const ix2 = Math.min(inner[2], outer[2])
  const iy2 = Math.min(inner[3], outer[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  return ((ix2 - ix1) * (iy2 - iy1)) / bboxArea(inner)
}

function bboxArea(b: BBox): number {
  return Math.max(1, (b[2] - b[0]) * (b[3] - b[1]))
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

  // Missing text shows up as an ASYMMETRIC gap along the axis lines stack on:
  // the OCR text hugs one edge (small near-gap) and leaves room for more lines on
  // the opposite, reading-continuation edge (large far-gap). Horizontal text
  // stacks top→bottom, so check the vertical gaps; vertical (tategaki) text
  // stacks columns right→left, so check the horizontal gaps. Symmetric margins
  // (text centred with room on BOTH sides) are just a short line in a big bubble,
  // not missing text — keeping them out of recovery avoids a full-canvas decode
  // plus crop re-OCR on every short-dialogue-in-large-bubble.
  const vertical = members.filter(member => member.textDirection === 'vertical').length * 2 >= members.length
  const nearGap = vertical ? Math.min(gapLeft, gapRight) : Math.min(gapTop, gapBottom)
  const farGap = vertical ? Math.max(gapLeft, gapRight) : Math.max(gapTop, gapBottom)
  if (nearGap <= threshold && farGap > threshold) {
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
  const transformed = recognized.blocks.map(block =>
    transformBlock(block, csx1, csy1, upscale, captureScale),
  )
  return cleanRecoveredBlocks(transformed)
}

/**
 * One balloon's crop should yield one coherent text block. Drop artefact blocks
 * (tiny / decoration / hallucinated-huge) the coarse-pass filter never saw, then
 * drop any smaller block that sits inside a larger sibling — a faint stray line
 * the upscaled re-OCR invented over the real text.
 */
function cleanRecoveredBlocks(blocks: readonly TextBlock[]): TextBlock[] {
  const real = blocks.filter(block => !isArtifactBlock(block))
  const byAreaDesc = [...real].sort((a, b) => bboxArea(b.bbox) - bboxArea(a.bbox))
  const kept: TextBlock[] = []
  for (const block of byAreaDesc) {
    if (kept.some(k => containment(block.bbox, k.bbox) >= RECOVERED_SUBBLOCK_CONTAINMENT)) continue
    kept.push(block)
  }
  return kept
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

// ── Splice: drop everything the recovered crop supersedes, append recovered ──

function splice(
  blocks: readonly TextBlock[],
  diagnoses: readonly Diagnosis[],
  recovered: readonly TextBlock[][],
): TextBlock[] {
  const drop = new Set<number>()
  for (let i = 0; i < diagnoses.length; i += 1) {
    // Lens returned nothing — keep the originals as-is.
    if (!recovered[i]!.length) continue
    // The crop re-OCR is authoritative for its whole anchor region, so drop every
    // pre-existing block mostly inside it — not just the centre-inside members —
    // to avoid leaving an overlapping coarse block beside the recovered text.
    const anchorBox = diagnoses[i]!.anchor.bbox
    for (let index = 0; index < blocks.length; index += 1) {
      if (containment(blocks[index]!.bbox, anchorBox) >= ANCHOR_REPLACE_CONTAINMENT) drop.add(index)
    }
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
