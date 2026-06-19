import type { BBox, Polygon } from '../domain/geometry'
import type { ChapterContentLayout, ChapterContentPage, ChapterContentRect } from '../domain/chapterContent'
import type { RecognizedTextPage, TextBlock, TextLine, TextWord } from '../domain/text'

export interface SourcePageSize {
  readonly width: number
  readonly height: number
}

export interface ChapterOcrChunk {
  readonly index: number
  readonly contentRect: ChapterContentRect
  readonly coreRect: ChapterContentRect
}

const MAX_CONTENT_WIDTH = 1280
const CHUNK_HEIGHT_PX = 1800
const CHUNK_OVERLAP_PX = 600

export function buildChapterContentLayout(pageSizes: readonly SourcePageSize[]): ChapterContentLayout {
  const maxWidth = Math.max(1, ...pageSizes.map(size => size.width))
  const contentWidth = Math.min(MAX_CONTENT_WIDTH, maxWidth)
  const pages: ChapterContentPage[] = []
  let y = 0

  pageSizes.forEach((sourceSize, pageIndex) => {
    const scale = contentWidth / Math.max(1, sourceSize.width)
    const height = sourceSize.height * scale
    pages.push({
      pageIndex,
      sourceSize,
      contentRect: { x: 0, y, width: contentWidth, height },
    })
    y += height
  })

  return { contentSize: { width: contentWidth, height: y }, pages }
}

export function chapterOcrChunks(layout: ChapterContentLayout): readonly ChapterOcrChunk[] {
  const chunks: ChapterOcrChunk[] = []
  const height = layout.contentSize.height
  const width = layout.contentSize.width
  const step = Math.max(1, CHUNK_HEIGHT_PX - CHUNK_OVERLAP_PX)
  let y = 0
  let index = 0

  while (y < height) {
    const bottom = Math.min(height, y + CHUNK_HEIGHT_PX)
    const hasPrevious = y > 0
    const hasNext = bottom < height
    const coreTop = hasPrevious ? y + CHUNK_OVERLAP_PX / 2 : y
    const coreBottom = hasNext ? bottom - CHUNK_OVERLAP_PX / 2 : bottom
    chunks.push({
      index,
      contentRect: { x: 0, y, width, height: bottom - y },
      coreRect: { x: 0, y: coreTop, width, height: Math.max(0, coreBottom - coreTop) },
    })
    if (bottom >= height) break
    y += step
    index += 1
  }

  return chunks
}

export function pagesIntersectingChunk(
  layout: ChapterContentLayout,
  chunk: ChapterOcrChunk,
): readonly ChapterContentPage[] {
  return layout.pages.filter(page => rectIntersection(page.contentRect, chunk.contentRect) !== null)
}

export function mapChunkRecognitionToChapter(
  recognized: RecognizedTextPage,
  chunk: ChapterOcrChunk,
): RecognizedTextPage {
  return {
    ...recognized,
    pageIndex: chunk.index,
    pageSize: [chunk.contentRect.width, Number.POSITIVE_INFINITY],
    blocks: recognized.blocks.map(block => mapBlock(block, chunk.contentRect.x, chunk.contentRect.y)),
  }
}

export function mergeChunkRecognitions(
  chunks: readonly RecognizedTextPage[],
  layout: ChapterContentLayout,
): RecognizedTextPage {
  const blocks = dedupeBlocks(chunks.flatMap(chunk => chunk.blocks))
  return {
    pageIndex: 0,
    pageSize: [layout.contentSize.width, layout.contentSize.height],
    detectedLanguage: chunks.find(chunk => chunk.detectedLanguage)?.detectedLanguage ?? null,
    blocks,
    timingMs: mergeTiming(chunks),
  }
}

export function rectIntersection(a: ChapterContentRect, b: ChapterContentRect): ChapterContentRect | null {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.width, b.x + b.width)
  const y2 = Math.min(a.y + a.height, b.y + b.height)
  return x1 < x2 && y1 < y2 ? { x: x1, y: y1, width: x2 - x1, height: y2 - y1 } : null
}

function mapBlock(block: TextBlock, dx: number, dy: number): TextBlock {
  return {
    ...block,
    bbox: translateBBox(block.bbox, dx, dy),
    polygon: translatePolygon(block.polygon, dx, dy),
    lines: block.lines.map(line => mapLine(line, dx, dy)),
    words: block.words.map(word => mapWord(word, dx, dy)),
  }
}

function mapLine(line: TextLine, dx: number, dy: number): TextLine {
  return {
    ...line,
    bbox: translateBBox(line.bbox, dx, dy),
    words: line.words.map(word => mapWord(word, dx, dy)),
  }
}

function mapWord(word: TextWord, dx: number, dy: number): TextWord {
  return { ...word, bbox: translateBBox(word.bbox, dx, dy) }
}

function translateBBox(bbox: BBox, dx: number, dy: number): BBox {
  return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]
}

function translatePolygon(polygon: Polygon, dx: number, dy: number): Polygon {
  return polygon.map(point => [point[0] + dx, point[1] + dy])
}

function dedupeBlocks(blocks: readonly TextBlock[]): TextBlock[] {
  const out: TextBlock[] = []
  for (const block of [...blocks].sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0])) {
    const duplicateIndex = out.findIndex(candidate => sameBlock(candidate, block))
    if (duplicateIndex === -1) out.push(block)
    else if (blockScore(block) > blockScore(out[duplicateIndex]!)) out[duplicateIndex] = block
  }
  return out.sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0])
}

function sameBlock(a: TextBlock, b: TextBlock): boolean {
  const sourceA = normalizeText(a.text)
  const sourceB = normalizeText(b.text)
  if (!sourceA || !sourceB) return false
  const related = sourceA.includes(sourceB) || sourceB.includes(sourceA)
  if (!related && sourceA !== sourceB) return false
  return bboxOverlapRatio(a.bbox, b.bbox) >= 0.65 || centerDistanceRatio(a.bbox, b.bbox) <= 0.25
}

function blockScore(block: TextBlock): number {
  return normalizeText(block.text).length * 4 + block.lines.length * 2 + bboxArea(block.bbox) / 10000
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
}

function bboxOverlapRatio(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  return ((ix2 - ix1) * (iy2 - iy1)) / Math.max(1, Math.min(bboxArea(a), bboxArea(b)))
}

function centerDistanceRatio(a: BBox, b: BBox): number {
  const acx = (a[0] + a[2]) / 2
  const acy = (a[1] + a[3]) / 2
  const bcx = (b[0] + b[2]) / 2
  const bcy = (b[1] + b[3]) / 2
  const scale = Math.max(1, bboxWidth(a), bboxHeight(a), bboxWidth(b), bboxHeight(b))
  return Math.hypot(acx - bcx, acy - bcy) / scale
}

function bboxArea(bbox: BBox): number {
  return bboxWidth(bbox) * bboxHeight(bbox)
}

function bboxWidth(bbox: BBox): number {
  return Math.max(0, bbox[2] - bbox[0])
}

function bboxHeight(bbox: BBox): number {
  return Math.max(0, bbox[3] - bbox[1])
}

function mergeTiming(chunks: readonly RecognizedTextPage[]): Record<string, number> {
  return chunks.reduce<Record<string, number>>((out, chunk) => {
    for (const [key, value] of Object.entries(chunk.timingMs)) out[key] = (out[key] ?? 0) + value
    return out
  }, {})
}
