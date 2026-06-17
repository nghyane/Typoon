import type { BBox, Point, Polygon } from '../domain/geometry'
import type { FontHint, TextPlacement, TextRole } from '../domain/planning'
import type { TextRegion } from '../domain/regions'
import type { RecognizedTextPage, TextBlock, TextUnit } from '../domain/text'

interface Anchor {
  readonly kind: TextRegion['kind']
  readonly bbox: BBox
  readonly confidence: number
  readonly innerBBox: BBox | null
}

export function textPlacementsFromRecognition(recognized: RecognizedTextPage, units: readonly TextUnit[]): TextPlacement[] {
  return blocksToTextPlacements(recognized.blocks, units, recognized.pageIndex, recognized.pageSize, [])
}

export function layoutPlacementsFromRegions(
  recognized: RecognizedTextPage,
  units: readonly TextUnit[],
  regions: readonly TextRegion[],
): TextPlacement[] {
  return blocksToTextPlacements(recognized.blocks, units, recognized.pageIndex, recognized.pageSize, regions)
}

function blocksToTextPlacements(
  blocks: readonly TextBlock[],
  units: readonly TextUnit[],
  pageIndex: number,
  pageSize: readonly [number, number],
  regions: readonly TextRegion[],
): TextPlacement[] {
  if (regions.length) return groupedTextPlacements(blocks, units, regions, pageIndex, pageSize)
  return textOnlyPlacements(blocks, units, pageIndex, pageSize)
}

function textOnlyPlacements(blocks: readonly TextBlock[], units: readonly TextUnit[], pageIndex: number, pageSize: readonly [number, number]): TextPlacement[] {
  return blocks
    .map((block, blockIndex) => ({ block, blockIndex }))
    .sort((a, b) => a.block.bbox[1] - b.block.bbox[1] || a.block.bbox[0] - b.block.bbox[0])
    .map(({ block, blockIndex }, index) => blockToTextPlacement(block, units[blockIndex]?.id ?? blockUnitId(pageIndex, blockIndex), index, pageIndex, pageSize))
}

function groupedTextPlacements(
  blocks: readonly TextBlock[],
  units: readonly TextUnit[],
  regions: readonly TextRegion[],
  pageIndex: number,
  pageSize: readonly [number, number],
): TextPlacement[] {
  const anchors = dedupeAnchors(regions).sort((a, b) => area(a.bbox) - area(b.bbox))
  const assigned = new Set<number>()
  const placements: TextPlacement[] = []

  for (const anchor of anchors) {
    const memberIds = blocks
      .map((block, index) => ({ block, index }))
      .filter(({ block, index }) => !assigned.has(index) && containsCenter(anchor.bbox, block.bbox))
      .map(({ index }) => index)
    if (!memberIds.length) continue
    const members = memberIds.map(index => ({ block: blocks[index]!, index }))
    placements.push(groupToTextPlacement(members, units, anchor, placements.length, pageIndex, pageSize))
    memberIds.forEach(index => assigned.add(index))
  }

  for (let i = 0; i < blocks.length; i++) {
    if (assigned.has(i)) continue
    placements.push(blockToTextPlacement(blocks[i]!, units[i]?.id ?? blockUnitId(pageIndex, i), placements.length, pageIndex, pageSize))
  }
  return placements.sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0])
}

function groupToTextPlacement(
  members: readonly { readonly block: TextBlock; readonly index: number }[],
  units: readonly TextUnit[],
  anchor: Anchor,
  index: number,
  pageIndex: number,
  pageSize: readonly [number, number],
): TextPlacement {
  const ordered = sortForReading(members.map(member => member.block))
  const orderedUnitIds = ordered.map(block => {
    const member = members.find(item => item.block === block)!
    return units[member.index]?.id ?? blockUnitId(pageIndex, member.index)
  })
  const inferred = classifyMergedBlocks(ordered)
  const role = anchor.kind === 'bubble' || anchor.kind === 'text_bubble' ? 'dialogue' : inferred
  const drawable = drawableForGroup(ordered, anchor, role, pageSize)
  return {
    id: `p${pageIndex}-r${index}`,
    pageIndex,
    pageSize,
    sourceUnitIds: orderedUnitIds,
    drawable,
    bbox: polygonBBox(drawable, pageSize),
    textBoxes: ordered.flatMap(block => block.lines.length ? block.lines.map(line => line.bbox) : [block.bbox]),
    role,
    rotationDeg: maxAbsRotation(ordered),
    confidence: Math.max(...ordered.map(block => block.confidence), anchor.confidence),
    fontHint: mergedFontHint(ordered),
  }
}

function blockToTextPlacement(block: TextBlock, unitId: string, index: number, pageIndex: number, pageSize: readonly [number, number]): TextPlacement {
  const role = classifyBlock(block)
  const drawable = drawableForBlock(block, role, pageSize)
  return {
    id: `p${pageIndex}-r${index}`,
    pageIndex,
    pageSize,
    sourceUnitIds: [unitId],
    drawable,
    bbox: polygonBBox(drawable, pageSize),
    textBoxes: block.lines.length ? block.lines.map(line => line.bbox) : [block.bbox],
    role,
    rotationDeg: block.rotationDeg,
    confidence: block.confidence,
    fontHint: fontHint(block),
  }
}

function blockUnitId(pageIndex: number, blockIndex: number): string {
  return `p${pageIndex}-b${blockIndex}`
}

function dedupeAnchors(regions: readonly TextRegion[]): Anchor[] {
  const candidates = regions.filter(region => region.kind === 'bubble' || region.kind === 'text_bubble' || region.kind === 'text_free')
  const clusters = clusterRegions(candidates, 0.7)
  const textBubbles = regions.filter(region => region.kind === 'text_bubble')
  const anchors: Anchor[] = []
  for (const cluster of clusters) {
    const winner = pickAnchor(cluster)
    if (!winner) continue
    const innerBBox = winner.kind === 'bubble'
      ? bestInnerRect(winner.bbox, textBubbles)
      : winner.kind === 'text_bubble'
        ? winner.bbox
        : null
    anchors.push({ ...winner, innerBBox })
  }
  return anchors
}

function clusterRegions(regions: readonly TextRegion[], iouThreshold: number): TextRegion[][] {
  const clusters: TextRegion[][] = []
  for (const region of regions) {
    const cluster = clusters.find(items => items.some(item => iou(item.bbox, region.bbox) > iouThreshold))
    if (cluster) cluster.push(region)
    else clusters.push([region])
  }
  return clusters
}

function pickAnchor(regions: readonly TextRegion[]): TextRegion | null {
  for (const kind of ['text_free', 'bubble', 'text_bubble'] as const) {
    const same = regions.filter(region => region.kind === kind)
    if (same.length) return same.reduce((best, region) => region.confidence > best.confidence ? region : best)
  }
  return null
}

function bestInnerRect(bubble: BBox, candidates: readonly TextRegion[]): BBox | null {
  const matches = candidates.filter(candidate => containment(candidate.bbox, bubble) >= 0.8)
  if (!matches.length) return null
  return matches.reduce((best, candidate) => candidate.confidence > best.confidence ? candidate : best).bbox
}

function drawableForGroup(
  members: readonly TextBlock[],
  anchor: Anchor,
  role: TextRole,
  pageSize: readonly [number, number],
): Polygon {
  if (anchor.kind === 'bubble' || anchor.kind === 'text_bubble') {
    const merged = anchor.innerBBox ? unionBBoxes([anchor.innerBBox, wordUnion(members)]) ?? anchor.innerBBox : anchor.bbox
    return ellipsePolygon(clipBBox(merged, pageSize))
  }
  if (role === 'sfx' && members.length === 1) return drawableForBlock(members[0]!, role, pageSize)
  return bboxToPolygon(clipBBox(unionBBoxes(members.map(block => block.bbox)) ?? anchor.bbox, pageSize))
}

function sortForReading(blocks: readonly TextBlock[]): TextBlock[] {
  const vertical = blocks.filter(block => block.textDirection === 'vertical').length
  if (vertical * 2 >= blocks.length) return [...blocks].sort((a, b) => b.bbox[0] - a.bbox[0] || a.bbox[1] - b.bbox[1])
  return [...blocks].sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0])
}

function classifyMergedBlocks(blocks: readonly TextBlock[]): TextRole {
  if (blocks.some(block => classifyBlock(block) === 'sfx')) return 'sfx'
  const text = blocks.map(block => block.text).join('\n')
  const chars = [...text].filter(ch => !/\s/u.test(ch)).length
  return chars > 30 ? 'narration' : 'dialogue'
}

function maxAbsRotation(blocks: readonly TextBlock[]): number {
  if (!blocks.length) return 0
  return blocks.reduce((best, block) => Math.abs(block.rotationDeg) > Math.abs(best) ? block.rotationDeg : best, 0)
}

function mergedFontHint(blocks: readonly TextBlock[]): FontHint | null {
  const hints = blocks.map(fontHint).filter((hint): hint is FontHint => hint !== null)
  if (!hints.length) return null
  const sourceFontPx = Math.round(median(hints.map(hint => hint.sourceFontPx ?? 0).filter(Boolean)))
  const sourceLineCount = hints.reduce((total, hint) => total + (hint.sourceLineCount ?? 1), 0)
  const chars = blocks.reduce((total, block) => total + [...block.text].filter(ch => !/\s/u.test(ch)).length, 0)
  const vertical = hints.filter(hint => hint.sourceDirection === 'vertical').length
  return {
    sourceFontPx,
    sourceLineCount,
    sourceAvgCharsPerLine: chars / Math.max(1, sourceLineCount),
    sourceDirection: vertical * 2 >= hints.length ? 'vertical' : 'horizontal',
  }
}

function wordUnion(blocks: readonly TextBlock[]): BBox {
  return unionBBoxes(blocks.flatMap(block => block.words.map(word => word.bbox)))
    ?? unionBBoxes(blocks.map(block => block.bbox))
    ?? [0, 0, 1, 1]
}

function containsCenter(outer: BBox, inner: BBox): boolean {
  const cx = (inner[0] + inner[2]) / 2
  const cy = (inner[1] + inner[3]) / 2
  return outer[0] <= cx && cx <= outer[2] && outer[1] <= cy && cy <= outer[3]
}

function containment(inner: BBox, outer: BBox): number {
  const ix1 = Math.max(inner[0], outer[0])
  const iy1 = Math.max(inner[1], outer[1])
  const ix2 = Math.min(inner[2], outer[2])
  const iy2 = Math.min(inner[3], outer[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  return ((ix2 - ix1) * (iy2 - iy1)) / area(inner)
}

function ellipsePolygon(bbox: BBox, vertices = 24): Polygon {
  const [x1, y1, x2, y2] = bbox
  const cx = (x1 + x2) / 2
  const cy = (y1 + y2) / 2
  const rx = (x2 - x1) / 2
  const ry = (y2 - y1) / 2
  const out: Point[] = []
  for (let i = 0; i < vertices; i++) {
    const angle = Math.PI * 2 * i / vertices
    out.push([cx + rx * Math.cos(angle), cy + ry * Math.sin(angle)])
  }
  return out
}

function classifyBlock(block: TextBlock): TextRole {
  const chars = [...block.text].filter(ch => !/\s/u.test(ch)).length
  const w = Math.max(1, block.bbox[2] - block.bbox[0])
  const h = Math.max(1, block.bbox[3] - block.bbox[1])
  if (Math.abs(block.rotationDeg) > 5) return 'sfx'
  if (chars <= 10 && w / h >= 1.4) return 'sfx'
  if (chars > 30) return 'narration'
  return 'dialogue'
}

function drawableForBlock(block: TextBlock, role: TextRole, pageSize: readonly [number, number]): Polygon {
  const glyph = medianGlyphSize(block) || 10
  const pad = Math.max(4, Math.round(glyph * (role === 'sfx' ? 0.08 : 0.20)))
  const unionBox = unionBBoxes(block.words.length ? block.words.map(word => word.bbox) : block.lines.map(line => line.bbox)) ?? block.bbox
  const x1 = unionBox[0] - pad
  const y1 = unionBox[1] - pad
  const x2 = unionBox[2] + pad
  const y2 = unionBox[3] + pad
  if (Math.abs(block.rotationDeg) > 1) {
    const cx = (x1 + x2) / 2
    const cy = (y1 + y2) / 2
    return orientedRect(cx, cy, x2 - x1, y2 - y1, block.rotationDeg)
  }
  return bboxToPolygon(clipBBox([x1, y1, x2, y2], pageSize))
}

function fontHint(block: TextBlock): FontHint | null {
  const samples = block.words.map(word => Math.min(word.bbox[2] - word.bbox[0], word.bbox[3] - word.bbox[1])).filter(n => n > 0)
  if (!samples.length) return null
  const lineCount = Math.max(1, block.lines.length)
  const chars = [...block.text].filter(ch => !/\s/u.test(ch)).length
  return {
    sourceFontPx: Math.round(median(samples)),
    sourceLineCount: lineCount,
    sourceAvgCharsPerLine: chars / lineCount,
    sourceDirection: block.textDirection,
  }
}

function orientedRect(cx: number, cy: number, w: number, h: number, rotationDeg: number): Polygon {
  const rad = rotationDeg * Math.PI / 180
  const cos = Math.cos(rad)
  const sin = Math.sin(rad)
  return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]].map(([x, y]) => [cx + x * cos - y * sin, cy + x * sin + y * cos] as Point)
}

function polygonBBox(polygon: Polygon, pageSize: readonly [number, number]): BBox {
  const xs = polygon.map(p => p[0])
  const ys = polygon.map(p => p[1])
  return clipBBox([Math.floor(Math.min(...xs)), Math.floor(Math.min(...ys)), Math.ceil(Math.max(...xs)), Math.ceil(Math.max(...ys))], pageSize)
}

function clipBBox(bbox: BBox, pageSize: readonly [number, number]): BBox {
  return [Math.max(0, bbox[0]), Math.max(0, bbox[1]), Math.min(pageSize[0], bbox[2]), Math.min(pageSize[1], bbox[3])]
}

function bboxToPolygon(b: BBox): Polygon {
  return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]
}

function unionBBoxes(boxes: readonly BBox[]): BBox | null {
  if (!boxes.length) return null
  return [Math.min(...boxes.map(b => b[0])), Math.min(...boxes.map(b => b[1])), Math.max(...boxes.map(b => b[2])), Math.max(...boxes.map(b => b[3]))]
}

function medianGlyphSize(block: TextBlock): number {
  const samples = block.lines.map(line => Math.min(line.bbox[2] - line.bbox[0], line.bbox[3] - line.bbox[1])).filter(n => n > 0)
  return samples.length ? median(samples) : 0
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}

function iou(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  return inter / (area(a) + area(b) - inter)
}

function area(b: BBox): number {
  return Math.max(1, (b[2] - b[0]) * (b[3] - b[1]))
}
