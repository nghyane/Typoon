// pipeline/ocrArtifacts.ts — drop OCR block artefacts before grouping.
//
// Ported from the canonical lens detector `filters.apply` (typoon-vision):
// removes decoration-only paragraphs, tiny / hallucinated-huge bboxes, and
// cross-column union artefacts that the tiled OCR pass emits near tile
// overlaps. This block hygiene is what keeps grouping robust independent of the
// region detector — a phantom paragraph that fuses several tategaki columns (or
// a hallucination over a screentone/art region) would otherwise survive into
// grouping as one giant placement covering the page.
//
// Thresholds are invariants tuned upstream; do not retune without the same
// fixture survey. Ratio-based predicates (huge_bbox, cross_column) are scale
// invariant, so they hold on the downscaled capture canvas too.

import type { BBox } from '../domain/geometry'
import type { RecognizedTextPage, TextBlock } from '../domain/text'

const MIN_BBOX_W = 25
const MIN_BBOX_H = 18
const MIN_BBOX_AREA = 700
// area / (glyph_short² × chars) — unit-less packing density. Real text packs
// glyphs ~1.5×; hallucinations on art emit huge bboxes with tiny glyphs.
const MAX_AREA_PER_GLYPH_RATIO = 6
// Absolute fallback when no usable line geometry exists.
const MAX_AREA_PER_CHAR_FALLBACK = 60000
const DECORATION_CHARS = new Set([...'★☆●○◎◇◆□■▲△▽▼※・…—–-_=+×÷'])
// A phantom paragraph whose lines geometrically sit inside ≥ this many other
// paragraphs is a tile-overlap fusion of distinct columns.
const CROSS_COLUMN_MIN_LINES_ABSORBED = 2
const CROSS_COLUMN_LINE_INSIDE_RATIO = 0.7

export function removeOcrArtifactBlocks(recognized: RecognizedTextPage): RecognizedTextPage {
  const kept = recognized.blocks.filter(block => !isArtifactBlock(block))
  const final = dropCrossColumnArtifacts(kept)
  if (final.length === recognized.blocks.length) return recognized
  return { ...recognized, blocks: final }
}

function isArtifactBlock(block: TextBlock): boolean {
  return bboxTooSmall(block.bbox)
    || isDecorationOnly(block.text)
    || bboxTooLargeForText(block.bbox, block.text, block.lines)
}

function bboxTooSmall(bbox: BBox): boolean {
  const w = bbox[2] - bbox[0]
  const h = bbox[3] - bbox[1]
  return w < MIN_BBOX_W || h < MIN_BBOX_H || w * h < MIN_BBOX_AREA
}

function isDecorationOnly(text: string): boolean {
  return ![...text].some(isLetterOrDigit)
}

function isLetterOrDigit(ch: string): boolean {
  if (DECORATION_CHARS.has(ch) || /\s/u.test(ch)) return false
  return /[\p{L}\p{N}]/u.test(ch)
}

function bboxTooLargeForText(bbox: BBox, text: string, lines: TextBlock['lines']): boolean {
  const area = Math.max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
  const chars = Math.max(1, countNonSpace(text))
  const glyphShorts = lines
    .map(line => Math.min(line.bbox[2] - line.bbox[0], line.bbox[3] - line.bbox[1]))
    .filter(n => n > 0)
  if (glyphShorts.length) {
    const glyph = median(glyphShorts)
    if (glyph > 0) return area / (glyph * glyph * chars) > MAX_AREA_PER_GLYPH_RATIO
  }
  return area / chars > MAX_AREA_PER_CHAR_FALLBACK
}

function dropCrossColumnArtifacts(blocks: readonly TextBlock[]): TextBlock[] {
  if (blocks.length < 3) return [...blocks]
  return blocks.filter((block, i) => {
    if (block.lines.length < CROSS_COLUMN_MIN_LINES_ABSORBED) return true
    const absorbing = new Set<number>()
    for (const line of block.lines) {
      for (let j = 0; j < blocks.length; j += 1) {
        if (j === i) continue
        if (bboxInsideRatio(line.bbox, blocks[j]!.bbox) >= CROSS_COLUMN_LINE_INSIDE_RATIO) {
          absorbing.add(j)
          break
        }
      }
    }
    return absorbing.size < CROSS_COLUMN_MIN_LINES_ABSORBED
  })
}

function bboxInsideRatio(child: BBox, parent: BBox): number {
  const ix1 = Math.max(child[0], parent[0])
  const iy1 = Math.max(child[1], parent[1])
  const ix2 = Math.min(child[2], parent[2])
  const iy2 = Math.min(child[3], parent[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  const area = Math.max(1, (child[2] - child[0]) * (child[3] - child[1]))
  return inter / area
}

function countNonSpace(text: string): number {
  return [...text].filter(ch => !/\s/u.test(ch)).length
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}
