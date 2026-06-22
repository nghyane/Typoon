import type { BBox, Point, Polygon } from '../domain/geometry'
import type { FontHint, TextPlacement, TextRole } from '../domain/planning'
import type { TextRegion } from '../domain/regions'
import type { RecognizedTextPage, TextBlock, TextUnit } from '../domain/text'
import { defaultLayoutHint, withTextLayoutHints } from './textLayoutHints'
import { classifyTextBlockRole, textRoleContext, type TextRoleContext } from './textRole'

interface Anchor {
  readonly kind: TextRegion['kind']
  readonly bbox: BBox
  readonly confidence: number
  readonly innerBBox: BBox | null
}

const TEXT_FREE_BUBBLE_AREA_RATIO_MIN = 0.70
const TEXT_BUBBLE_CLUSTER_OVERLAP_MIN = 0.20

export function textPlacementsFromRecognition(recognized: RecognizedTextPage, units: readonly TextUnit[], roleContext?: TextRoleContext): TextPlacement[] {
  return blocksToTextPlacements(recognized.blocks, units, recognized.pageIndex, recognized.pageSize, [], roleContext)
}

export function layoutPlacementsFromRegions(
  recognized: RecognizedTextPage,
  units: readonly TextUnit[],
  regions: readonly TextRegion[],
  roleContext?: TextRoleContext,
): TextPlacement[] {
  return blocksToTextPlacements(recognized.blocks, units, recognized.pageIndex, recognized.pageSize, regions, roleContext)
}

function blocksToTextPlacements(
  blocks: readonly TextBlock[],
  units: readonly TextUnit[],
  pageIndex: number,
  pageSize: readonly [number, number],
  regions: readonly TextRegion[],
  roleContext?: TextRoleContext,
): TextPlacement[] {
  const ctx = roleContext ?? textRoleContext(blocks)
  const placements = regions.length
    ? groupedTextPlacements(blocks, units, regions, pageIndex, pageSize, ctx)
    : textOnlyPlacements(blocks, units, pageIndex, pageSize, ctx)
  return withTextLayoutHints(placements, pageSize)
}

function textOnlyPlacements(
  blocks: readonly TextBlock[],
  units: readonly TextUnit[],
  pageIndex: number,
  pageSize: readonly [number, number],
  roleContext: TextRoleContext,
): TextPlacement[] {
  return blocks
    .map((block, blockIndex) => ({ block, blockIndex }))
    .sort((a, b) => a.block.bbox[1] - b.block.bbox[1] || a.block.bbox[0] - b.block.bbox[0])
    .map(({ block, blockIndex }, index) => blockToTextPlacement(block, units[blockIndex]?.id ?? blockUnitId(pageIndex, blockIndex), index, pageIndex, pageSize, roleContext))
}

function groupedTextPlacements(
  blocks: readonly TextBlock[],
  units: readonly TextUnit[],
  regions: readonly TextRegion[],
  pageIndex: number,
  pageSize: readonly [number, number],
  roleContext: TextRoleContext,
): TextPlacement[] {
  const anchors = dedupeAnchors(regions).sort(compareAnchors)
  const assigned = new Set<number>()
  const placements: TextPlacement[] = []

  for (const anchor of anchors) {
    const memberIds = blocks
      .map((block, index) => ({ block, index }))
      .filter(({ block, index }) => !assigned.has(index) && containsCenter(anchor.bbox, block.bbox))
      .map(({ index }) => index)
    if (!memberIds.length) continue
    // A bubble anchor is a single semantic unit — do not sub-group OCR
    // blocks inside it.  Subgrouping is only meaningful for free-text
    // regions that may span multiple independent text areas.
    const subgroups = anchor.kind === 'bubble' || anchor.kind === 'text_bubble'
      ? [Array.from(memberIds)]
      : subgroupBlockIds(memberIds, blocks, anchor.innerBBox ?? anchor.bbox)
    for (const subgroup of splitMixedRoleSubgroups(subgroups, blocks, roleContext)) {
      const members = subgroup.map(index => ({ block: blocks[index]!, index }))
      placements.push(groupToTextPlacement(members, units, anchor, placements.length, pageIndex, pageSize, roleContext))
    }
    memberIds.forEach(index => assigned.add(index))
  }

  for (let i = 0; i < blocks.length; i++) {
    if (assigned.has(i)) continue
    placements.push(blockToTextPlacement(blocks[i]!, units[i]?.id ?? blockUnitId(pageIndex, i), placements.length, pageIndex, pageSize, roleContext))
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
  roleContext: TextRoleContext,
): TextPlacement {
  const ordered = sortForReading(members.map(member => member.block))
  const orderedUnitIds = ordered.map(block => {
    const member = members.find(item => item.block === block)!
    return units[member.index]?.id ?? blockUnitId(pageIndex, member.index)
  })
  const inferred = classifyMergedBlocks(ordered, roleContext)
  const role = (anchor.kind === 'bubble' || anchor.kind === 'text_bubble') && inferred !== 'sfx' ? 'dialogue' : inferred
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
    layoutHint: defaultLayoutHint(),
  }
}

function blockToTextPlacement(block: TextBlock, unitId: string, index: number, pageIndex: number, pageSize: readonly [number, number], roleContext: TextRoleContext): TextPlacement {
  const role = classifyTextBlockRole(block, roleContext)
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
    layoutHint: defaultLayoutHint(),
  }
}

function blockUnitId(pageIndex: number, blockIndex: number): string {
  return `p${pageIndex}-b${blockIndex}`
}

function dedupeAnchors(regions: readonly TextRegion[]): Anchor[] {
  const candidates = regions.filter(region => region.kind === 'bubble' || region.kind === 'text_bubble' || region.kind === 'text_free')
  const textBubbles = candidates.filter(region => region.kind === 'text_bubble')
  return clusterRegions(candidates, 0.7)
    .map(cluster => pickAnchor(cluster, textBubbles))
    .filter((anchor): anchor is Anchor => anchor !== null)
}

function clusterRegions(regions: readonly TextRegion[], iouThreshold: number): TextRegion[][] {
  const clusters: TextRegion[][] = []
  for (const region of regions) {
    const cluster = clusters.find(items => items.some(item => sameAnchorCluster(item, region, iouThreshold)))
    if (cluster) cluster.push(region)
    else clusters.push([region])
  }
  return clusters
}

function sameAnchorCluster(a: TextRegion, b: TextRegion, iouThreshold: number): boolean {
  if (iou(a.bbox, b.bbox) > iouThreshold) return true
  if (a.kind === 'text_bubble' && b.kind === 'text_bubble') return overlapFraction(a.bbox, b.bbox) >= TEXT_BUBBLE_CLUSTER_OVERLAP_MIN
  if (!isBubblePair(a, b)) return false
  const inner = a.kind === 'bubble' ? b : a
  const outer = a.kind === 'bubble' ? a : b
  return containment(inner.bbox, outer.bbox) >= 0.65 || containsCenter(outer.bbox, inner.bbox)
}

function isBubblePair(a: TextRegion, b: TextRegion): boolean {
  return (a.kind === 'bubble' && (b.kind === 'text_bubble' || b.kind === 'text_free'))
    || (b.kind === 'bubble' && (a.kind === 'text_bubble' || a.kind === 'text_free'))
}

function overlapFraction(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  return ((ix2 - ix1) * (iy2 - iy1)) / Math.max(1, Math.min(area(a), area(b)))
}

function pickAnchor(cluster: readonly TextRegion[], textBubbles: readonly TextRegion[]): Anchor | null {
  const winner = pickAnchorRegion(cluster)
  if (!winner) return null
  if (winner.kind === 'bubble') return { ...winner, innerBBox: bestInnerBBox(winner.bbox, textBubbles) }
  if (winner.kind === 'text_bubble') {
    const bbox = unionBBoxes(cluster.filter(region => region.kind === 'text_bubble').map(region => region.bbox)) ?? winner.bbox
    return { ...winner, bbox, innerBBox: bbox }
  }
  return { ...winner, innerBBox: null }
}

function pickAnchorRegion(cluster: readonly TextRegion[]): TextRegion | null {
  const textFree = bestRegion(cluster, 'text_free')
  const bubble = bestRegion(cluster, 'bubble')
  if (textFree && (!bubble || area(textFree.bbox) / area(bubble.bbox) >= TEXT_FREE_BUBBLE_AREA_RATIO_MIN)) return textFree
  if (bubble) return bubble
  return bestRegion(cluster, 'text_bubble')
}

function bestRegion(cluster: readonly TextRegion[], kind: TextRegion['kind']): TextRegion | null {
  const sameKind = cluster.filter(region => region.kind === kind)
  if (!sameKind.length) return null
  return sameKind.reduce((best, region) => region.confidence > best.confidence ? region : best)
}

function bestInnerBBox(bubble: BBox, textBubbles: readonly TextRegion[]): BBox | null {
  const matches = textBubbles.filter(region => containment(region.bbox, bubble) >= 0.80)
  if (!matches.length) return null
  return matches.reduce((best, region) => region.confidence > best.confidence ? region : best).bbox
}

function compareAnchors(a: Anchor, b: Anchor): number {
  return area(a.bbox) - area(b.bbox)
}

function subgroupBlockIds(indices: readonly number[], blocks: readonly TextBlock[], container: BBox): number[][] {
  if (indices.length <= 1) return [Array.from(indices)]

  const boxes = indices.map(index => blocks[index]!.bbox)
  const textUnion = unionBBoxes(boxes) ?? container
  const heights = boxes.map(box => Math.max(1, box[3] - box[1]))
  const medH = median(heights)
  const sortedByY = [...boxes].sort((a, b) => a[1] - b[1])
  const gaps = sortedByY.slice(0, -1).map((box, i) => Math.max(0, sortedByY[i + 1]![1] - box[3]))
  const largeGaps = gaps.filter(gap => gap > medH * 1.25).length
  const allInside = indices.every(index => containsCenter(container, blocks[index]!.bbox))

  if (allInside && largeGaps === 0) return [Array.from(indices)]

  const cw = Math.max(1, container[2] - container[0])
  const ch = Math.max(1, container[3] - container[1])
  const uw = Math.max(1, textUnion[2] - textUnion[0])
  const uh = Math.max(1, textUnion[3] - textUnion[1])
  const strict = indices.length >= 5 && (uh / ch > 0.80 || largeGaps >= 2)
  if (!strict && indices.length <= 6 && uh / ch < 0.85 && uw / cw < 0.98 && largeGaps === 0) return [Array.from(indices)]

  const parent = new Map<number, number>()
  indices.forEach(index => parent.set(index, index))
  const find = (index: number): number => {
    let root = parent.get(index) ?? index
    while ((parent.get(root) ?? root) !== root) root = parent.get(root) ?? root
    let cur = index
    while ((parent.get(cur) ?? cur) !== root) {
      const next = parent.get(cur) ?? cur
      parent.set(cur, root)
      cur = next
    }
    return root
  }
  const join = (a: number, b: number): void => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent.set(rb, ra)
  }

  for (let aPos = 0; aPos < indices.length; aPos += 1) {
    const i = indices[aPos]!
    const blockA = blocks[i]!
    const a = blockA.bbox
    const ah = Math.max(1, a[3] - a[1])
    const aVert = blockA.textDirection === 'vertical'
    for (const j of indices.slice(aPos + 1)) {
      const blockB = blocks[j]!
      const b = blockB.bbox
      const bh = Math.max(1, b[3] - b[1])
      const bVert = blockB.textDirection === 'vertical'

      // Never join blocks of different direction (vertical + horizontal)
      if (aVert !== bVert) continue

      // Don't join blocks with significantly different font sizes.
      // Each size tier should be a separate placement to preserve hierarchy.
      const fontA = estimateBlockFontPx(blockA)
      const fontB = estimateBlockFontPx(blockB)
      if (fontA > 0 && fontB > 0) {
        const fontRatio = Math.max(fontA, fontB) / Math.min(fontA, fontB)
        if (fontRatio > 1.5) continue
      }

      const minH = Math.max(1, Math.min(ah, bh))
      const heightRatio = Math.max(ah, bh) / minH

      // Vertical (tategaki) columns: width-scaled x-gap (ported from lens_native.py)
      // Horizontal blocks: height-scaled gap (ported from groups.py subgroup_blocks)
      if (aVert) {
        const aw = Math.max(1, a[2] - a[0])
        const bw = Math.max(1, b[2] - b[0])
        const gapCap = Math.max(80, Math.min(aw, bw) * 2.0)
        const yOverlapRatio = yOverlap(a, b)
        const gapX = xGap(a, b)
        if (yOverlapRatio >= 0.55 && gapX <= gapCap && heightRatio < 1.8) join(i, j)
      } else {
        const sameColumn = xOverlap(a, b) > (strict ? 0.70 : 0.55) && yGap(a, b) <= minH * (strict ? 0.65 : 1.20) && heightRatio < (strict ? 1.7 : 2.1)
        const sameRow = yOverlap(a, b) > (strict ? 0.70 : 0.60) && xGap(a, b) <= minH * (strict ? 1.50 : 2.00) && heightRatio < (strict ? 1.7 : 2.1)
        const overlaps = !strict && (iou(a, b) > 0.12 || containment(a, b) > 0.35 || containment(b, a) > 0.35)
        if (sameColumn || sameRow || overlaps) join(i, j)
      }
    }
  }

  const groups = new Map<number, number[]>()
  for (const index of indices) {
    const root = find(index)
    const group = groups.get(root)
    if (group) group.push(index)
    else groups.set(root, [index])
  }
  return [...groups.values()].map(group => group.sort((a, b) => blocks[a]!.bbox[1] - blocks[b]!.bbox[1] || blocks[a]!.bbox[0] - blocks[b]!.bbox[0]))
}

function splitMixedRoleSubgroups(
  subgroups: readonly (readonly number[])[],
  blocks: readonly TextBlock[],
  roleContext: TextRoleContext,
): number[][] {
  return subgroups.flatMap(group => {
    const sfx: number[] = []
    const nonSfx: number[] = []
    for (const index of group) {
      if (classifyTextBlockRole(blocks[index]!, roleContext) === 'sfx') sfx.push(index)
      else nonSfx.push(index)
    }
    if (!sfx.length || !nonSfx.length) return [Array.from(group)]
    return [nonSfx, sfx].map(sortBlockIndices(blocks))
  })
}

function sortBlockIndices(blocks: readonly TextBlock[]): (indices: number[]) => number[] {
  return indices => indices.sort((a, b) => blocks[a]!.bbox[1] - blocks[b]!.bbox[1] || blocks[a]!.bbox[0] - blocks[b]!.bbox[0])
}

function drawableForGroup(
  members: readonly TextBlock[],
  _anchor: Anchor,
  role: TextRole,
  pageSize: readonly [number, number],
): Polygon {
  if (role === 'sfx' && members.length === 1) return drawableForBlock(members[0]!, role, pageSize)
  return drawableForMembers(members, role, pageSize)
}

function drawableForMembers(members: readonly TextBlock[], role: TextRole, pageSize: readonly [number, number]): Polygon {
  const glyph = median(members.map(medianGlyphSize).filter(n => n > 0)) || 10
  const pad = Math.max(4, Math.round(glyph * (role === 'sfx' ? 0.08 : 0.20)))
  const unionBox = wordUnion(members)
  const bbox: BBox = [unionBox[0] - pad, unionBox[1] - pad, unionBox[2] + pad, unionBox[3] + pad]
  const rotationDeg = maxAbsRotation(members)
  if (role === 'sfx' && Math.abs(rotationDeg) > 1) {
    const cx = (bbox[0] + bbox[2]) / 2
    const cy = (bbox[1] + bbox[3]) / 2
    return orientedRect(cx, cy, bbox[2] - bbox[0], bbox[3] - bbox[1], rotationDeg)
  }
  return bboxToPolygon(clipBBox(bbox, pageSize))
}

function sortForReading(blocks: readonly TextBlock[]): TextBlock[] {
  const vertical = blocks.filter(block => block.textDirection === 'vertical').length
  if (vertical * 2 >= blocks.length) return [...blocks].sort((a, b) => b.bbox[0] - a.bbox[0] || a.bbox[1] - b.bbox[1])
  return [...blocks].sort((a, b) => a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0])
}

function classifyMergedBlocks(blocks: readonly TextBlock[], roleContext: TextRoleContext): TextRole {
  if (blocks.some(block => classifyTextBlockRole(block, roleContext) === 'sfx')) return 'sfx'
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
  const sourceSamples = blocks.flatMap(sourceFontSamples)
  const sourceFontPx = Math.round(median(sourceSamples.length ? sourceSamples : hints.map(hint => hint.sourceFontPx ?? 0).filter(Boolean)))
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

function xOverlap(a: BBox, b: BBox): number {
  const overlap = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
  return overlap / Math.max(1, Math.min(a[2] - a[0], b[2] - b[0]))
}

function yOverlap(a: BBox, b: BBox): number {
  const overlap = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  return overlap / Math.max(1, Math.min(a[3] - a[1], b[3] - b[1]))
}

function xGap(a: BBox, b: BBox): number {
  return Math.max(0, Math.max(a[0], b[0]) - Math.min(a[2], b[2]))
}

function yGap(a: BBox, b: BBox): number {
  return Math.max(0, Math.max(a[1], b[1]) - Math.min(a[3], b[3]))
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
  const samples = sourceFontSamples(block)
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

function sourceFontSamples(block: TextBlock): number[] {
  const lineSamples = block.lines
    .map(line => block.textDirection === 'vertical' ? line.bbox[2] - line.bbox[0] : line.bbox[3] - line.bbox[1])
    .filter(n => n > 0)
  if (lineSamples.length) return lineSamples
  return block.words
    .map(word => Math.min(word.bbox[2] - word.bbox[0], word.bbox[3] - word.bbox[1]))
    .filter(n => n > 0)
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

function estimateBlockFontPx(block: TextBlock): number {
  if (block.lines.length) {
    const samples = block.lines
      .map(line => block.textDirection === 'vertical' ? line.bbox[2] - line.bbox[0] : line.bbox[3] - line.bbox[1])
      .filter(n => n > 0)
    if (samples.length) return median(samples)
  }
  if (block.words.length) {
    const samples = block.words.map(word => Math.min(word.bbox[2] - word.bbox[0], word.bbox[3] - word.bbox[1])).filter(n => n > 0)
    if (samples.length) return median(samples)
  }
  return 0
}

function area(b: BBox): number {
  return Math.max(1, (b[2] - b[0]) * (b[3] - b[1]))
}
