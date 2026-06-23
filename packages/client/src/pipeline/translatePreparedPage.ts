/** Build translation inputs/results for one PreparedPage. */

import type { Translator } from '../translators/translator'
import type { BBox, OrientedBox, Polygon } from '../domain/geometry'
import { orientedFromBBox, orientedPairFrame } from '../domain/geometry'
import type { TextRegion } from '../domain/regions'
import type { RecognizedTextPage, TextBlock, TextDirection, TextLine, TextUnit, TextWord } from '../domain/text'
import type { TranslationUnit, TranslatedUnit } from '../domain/translation'
import type { PreparedPageHandle } from '../domain/prepared'
import { textUnitsFromBlocks } from './textUnits'
import { translationUnitsFromTextUnits } from './translationUnits'
import { classifyTextBlockRole, textRoleContext, type TextRoleContext } from './textRole'

export interface PreparedTextResult {
  readonly recognized: RecognizedTextPage
  readonly textUnits: readonly TextUnit[]
  readonly translationUnits: readonly TranslationUnit[]
}

export interface PreparedTranslationResult extends PreparedTextResult {
  readonly translations: readonly TranslatedUnit[]
}

export function preparedTextFromRecognition(args: {
  readonly handle: PreparedPageHandle
  readonly recognized: RecognizedTextPage
}): PreparedTextResult {
  return textFromRecognition({ pageIndex: args.handle.preparedPageIndex, recognized: args.recognized })
}

export function textFromRecognition(args: {
  readonly pageIndex: number
  readonly recognized: RecognizedTextPage
  readonly regions?: readonly TextRegion[] | null
  readonly roleContext?: TextRoleContext
}): PreparedTextResult {
  const recognized = groupSpeechLines(args.recognized, args.regions ?? null, args.roleContext)
  const textUnits = textUnitsFromBlocks(
    recognized.blocks,
    args.pageIndex,
    args.roleContext,
  )
  const translationUnits = translationUnitsFromTextUnits(textUnits)

  return {
    recognized,
    textUnits,
    translationUnits,
  }
}

interface SpeechLine {
  readonly sourceBlock: TextBlock
  readonly bbox: BBox
  readonly text: string
  readonly rotationDeg: number
  readonly textDirection: TextDirection
  readonly confidence: number
  readonly line: TextLine
  readonly words: readonly TextWord[]
  /** True rotated extents when OCR provided them; else derived from bbox. */
  readonly oriented: OrientedBox
}

interface LineGeometryStats {
  readonly medianFontPx: number
  readonly fontScalePx: number
  readonly medianEdgeCost: number
  readonly edgeCostScale: number
  readonly maxEdgeCost: number
  readonly neighborCount: number
}

interface CandidateEdge {
  readonly a: number
  readonly b: number
  readonly cost: number
}

interface LinePairMetrics {
  readonly direction: TextDirection
  readonly localFontPx: number
  readonly primaryGapPx: number
  readonly primaryGapFont: number
  readonly secondaryOverlap: number
  readonly centerShiftPx: number
  readonly centerShiftFont: number
  readonly centerShiftSpan: number
  readonly fontRatio: number
  readonly angleSin: number
  readonly maxSecondarySpan: number
}

interface LineGeometryGraph {
  readonly stats: LineGeometryStats
  readonly edges: readonly CandidateEdge[]
}

const ROBUST_INLIER_Z = 2.5
const OCR_BOX_NOISE_FRACTION = 0.08
const GEOMETRY_TWIN_NOISE = ROBUST_INLIER_Z * OCR_BOX_NOISE_FRACTION
const MIN_TEXT_REGION_EDGE_COST = 2.35
const MAX_TEXT_REGION_EDGE_COST = 3.15

function groupSpeechLines(recognized: RecognizedTextPage, regions: readonly TextRegion[] | null, roleContext?: TextRoleContext): RecognizedTextPage {
  if (recognized.blocks.length <= 1 && recognized.blocks.every(block => block.lines.length <= 1)) return recognized
  const context = roleContext ?? textRoleContext(recognized.blocks)
  const ordered = dedupeSpeechLines(
    recognized.blocks
      .flatMap(block => speechLinesFromBlock(block))
      .sort(compareSpeechLinesForReading),
  )
  const regionKeys = ordered.map(line => speechLineRegionKey(line, regions))
  const graph = lineGeometryGraph(ordered)
  const parent = ordered.map((_, index) => index)
  const find = (index: number): number => {
    let root = parent[index] ?? index
    while ((parent[root] ?? root) !== root) root = parent[root] ?? root
    while ((parent[index] ?? index) !== root) {
      const next = parent[index] ?? index
      parent[index] = root
      index = next
    }
    return root
  }
  const join = (a: number, b: number): void => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent[rb] = ra
  }

  for (const edge of graph.edges) {
    const a = ordered[edge.a]!
    const b = ordered[edge.b]!
    const strongContinuation = strongLineContinuation(a, b, graph.stats)
    if (differentRegionContainers(regionKeys[edge.a] ?? null, regionKeys[edge.b] ?? null) && !strongContinuation) continue
    // Respect Lens paragraph boundaries — lines from different paragraphs
    // must not merge unless the geometry convincingly says they belong together.
    if (a.sourceBlock !== b.sourceBlock && !strongContinuation) continue
    if (!canMergeSpeechLine(a, context) || !canMergeSpeechLine(b, context)) continue
    if (!sameTextRegionPair(a, b, graph.stats)) continue
    if (hardNegativeLineConflict(a, b, graph.stats)) continue
    if (edge.cost <= graph.stats.maxEdgeCost) join(edge.a, edge.b)
  }

  const groupsByRoot = new Map<number, SpeechLine[]>()
  for (let i = 0; i < ordered.length; i++) {
    const root = find(i)
    const group = groupsByRoot.get(root)
    if (group) group.push(ordered[i]!)
    else groupsByRoot.set(root, [ordered[i]!])
  }
  const groups = [...groupsByRoot.values()]

  const blocks = groups.map(group => mergeSpeechLineGroup(group))
  return { ...recognized, blocks: blocks.sort(compareBlocksForReading) }
}

function speechLineRegionKey(line: SpeechLine, regions: readonly TextRegion[] | null): string | null {
  if (!regions?.length) return null
  const cx = centerX(line.bbox)
  const cy = centerY(line.bbox)
  const anchors = regions
    .map((region, index) => ({ region, index }))
    .filter(({ region }) => region.kind === 'text_bubble' || region.kind === 'bubble' || region.kind === 'text_free')
    .sort((a, b) => regionPriority(a.region.kind) - regionPriority(b.region.kind) || bboxArea(a.region.bbox) - bboxArea(b.region.bbox))
  const match = anchors.find(({ region }) => region.bbox[0] <= cx && cx <= region.bbox[2] && region.bbox[1] <= cy && cy <= region.bbox[3])
  return match ? `${match.region.kind}:${match.index}` : null
}

function differentRegionContainers(a: string | null, b: string | null): boolean {
  return isOuterBubbleKey(a) && isOuterBubbleKey(b) && a !== b
}

function isOuterBubbleKey(key: string | null): boolean {
  return key?.startsWith('bubble:') ?? false
}

function regionPriority(kind: TextRegion['kind']): number {
  if (kind === 'bubble') return 0
  if (kind === 'text_bubble') return 1
  return 2
}

function speechLinesFromBlock(block: TextBlock): SpeechLine[] {
  if (block.lines.length) {
    return block.lines.map(line => ({
      sourceBlock: block,
      bbox: line.bbox,
      text: line.text,
      rotationDeg: line.rotationDeg,
      textDirection: block.textDirection,
      confidence: block.confidence,
      line,
      words: line.words.length ? line.words : wordsInsideBBox(block.words, line.bbox),
      oriented: line.oriented ?? orientedFromBBox(line.bbox, line.rotationDeg),
    }))
  }

  const line: TextLine = {
    bbox: block.bbox,
    text: block.text,
    rotationDeg: block.rotationDeg,
    words: block.words,
  }
  return [{
    sourceBlock: block,
    bbox: block.bbox,
    text: block.text,
    rotationDeg: block.rotationDeg,
    textDirection: block.textDirection,
    confidence: block.confidence,
    line,
    words: block.words,
    oriented: orientedFromBBox(block.bbox, block.rotationDeg),
  }]
}

function wordsInsideBBox(words: readonly TextWord[], bbox: BBox): readonly TextWord[] {
  return words.filter(word => containsCenter(bbox, word.bbox))
}

function containsCenter(outer: BBox, inner: BBox): boolean {
  const cx = (inner[0] + inner[2]) / 2
  const cy = (inner[1] + inner[3]) / 2
  return outer[0] <= cx && cx <= outer[2] && outer[1] <= cy && cy <= outer[3]
}

function canMergeSpeechLine(line: SpeechLine, context: ReturnType<typeof textRoleContext>): boolean {
  if (!line.text.trim()) return false
  if (classifyTextBlockRole(line.sourceBlock, context) === 'sfx') return false
  return normalizedCharCount(line.text) <= 80
}

function dedupeSpeechLines(lines: readonly SpeechLine[]): SpeechLine[] {
  const out: SpeechLine[] = []
  for (const line of lines) {
    const duplicateIndex = out.findIndex(candidate => sameOcrLine(candidate, line))
    if (duplicateIndex === -1) {
      out.push(line)
      continue
    }
    if (speechLineScore(line) > speechLineScore(out[duplicateIndex]!)) out[duplicateIndex] = line
  }
  return out.sort(compareSpeechLinesForReading)
}

function sameOcrLine(a: SpeechLine, b: SpeechLine): boolean {
  if (a.textDirection !== b.textDirection) return false
  if (!sameGeometryBox(a.bbox, b.bbox)) return false
  const angleNoise = Math.abs(Math.sin((a.rotationDeg - b.rotationDeg) * Math.PI / 180))
  if (angleNoise > OCR_BOX_NOISE_FRACTION) return false
  return relatedTextForGrouping(a.text, b.text)
    || samePunctuationIntent(a.text, b.text)
    || normalizedCharCount(a.text) === normalizedCharCount(b.text)
}

function sameGeometryBox(a: BBox, b: BBox): boolean {
  const maxWidth = Math.max(1, bboxWidth(a), bboxWidth(b))
  const maxHeight = Math.max(1, bboxHeight(a), bboxHeight(b))
  const centerDistance = Math.hypot(centerX(a) - centerX(b), centerY(a) - centerY(b))
  const diagonal = Math.hypot(maxWidth, maxHeight)
  const sizeDelta = Math.max(
    Math.abs(bboxWidth(a) - bboxWidth(b)) / maxWidth,
    Math.abs(bboxHeight(a) - bboxHeight(b)) / maxHeight,
  )
  const overlap = intersectionArea(a, b) / Math.max(1, Math.min(bboxArea(a), bboxArea(b)))
  return centerDistance / Math.max(1, diagonal) <= GEOMETRY_TWIN_NOISE
    && sizeDelta <= GEOMETRY_TWIN_NOISE
    && overlap >= 1 - GEOMETRY_TWIN_NOISE
}

function speechLineScore(line: SpeechLine): number {
  return normalizedCharCount(line.text) * 4 + line.words.length + line.confidence
}

function lineFontPx(line: SpeechLine): number {
  return line.textDirection === 'vertical'
    ? Math.max(0, line.oriented.w)
    : Math.max(0, line.oriented.h)
}

function lineGeometryGraph(lines: readonly SpeechLine[]): LineGeometryGraph {
  const fonts = lines.map(lineFontPx).filter(value => value > 0)
  const medianFontPx = median(fonts) || median(lines.map(line => Math.min(bboxWidth(line.bbox), bboxHeight(line.bbox))).filter(value => value > 0)) || 16
  const fontScalePx = robustScale(fonts, medianFontPx) || medianFontPx * OCR_BOX_NOISE_FRACTION
  const neighborCount = Math.max(1, Math.ceil(Math.log2(lines.length + 1)))
  const nearestByLine: CandidateEdge[][] = lines.map(() => [])

  for (let i = 0; i < lines.length; i++) {
    for (let j = i + 1; j < lines.length; j++) {
      const cost = pairCost(lines[i]!, lines[j]!, { medianFontPx, fontScalePx })
      if (!Number.isFinite(cost)) continue
      const edge = { a: i, b: j, cost }
      addNearestEdge(nearestByLine[i]!, edge, neighborCount)
      addNearestEdge(nearestByLine[j]!, edge, neighborCount)
    }
  }

  const nearestCosts = nearestByLine.map(edges => edges[0]?.cost).filter((cost): cost is number => cost !== undefined)

  const medianEdgeCost = median(nearestCosts) || 0
  const edgeCostScale = robustScale(nearestCosts, medianEdgeCost) || Math.max(1, medianEdgeCost * OCR_BOX_NOISE_FRACTION)
  const stats = {
    medianFontPx,
    fontScalePx,
    medianEdgeCost,
    edgeCostScale,
    maxEdgeCost: clamp(medianEdgeCost + ROBUST_INLIER_Z * edgeCostScale, MIN_TEXT_REGION_EDGE_COST, MAX_TEXT_REGION_EDGE_COST),
    neighborCount,
  }
  return { stats, edges: uniqueEdges(nearestByLine).sort((a, b) => a.cost - b.cost) }
}

function addNearestEdge(edges: CandidateEdge[], edge: CandidateEdge, limit: number): void {
  edges.push(edge)
  edges.sort((a, b) => a.cost - b.cost)
  if (edges.length > limit) edges.length = limit
}

function uniqueEdges(groups: readonly CandidateEdge[][]): CandidateEdge[] {
  const byPair = new Map<string, CandidateEdge>()
  for (const edges of groups) {
    for (const edge of edges) {
      const a = Math.min(edge.a, edge.b)
      const b = Math.max(edge.a, edge.b)
      const key = `${a}:${b}`
      const current = byPair.get(key)
      if (!current || edge.cost < current.cost) byPair.set(key, { a, b, cost: edge.cost })
    }
  }
  return [...byPair.values()]
}

function pairCost(a: SpeechLine, b: SpeechLine, stats: Pick<LineGeometryStats, 'medianFontPx' | 'fontScalePx'>): number {
  const metrics = linePairMetrics(a, b, stats.medianFontPx)
  return metrics ? textRegionCost(metrics) : Infinity
}

function sameTextRegionPair(a: SpeechLine, b: SpeechLine, stats: LineGeometryStats): boolean {
  if (sameOcrLine(a, b)) return true
  const metrics = linePairMetrics(a, b, stats.medianFontPx)
  if (!metrics) return false
  if (metrics.angleSin > 0.34) return false
  if (metrics.primaryGapFont > maxPrimaryGapFont(metrics)) return false
  if (metrics.fontRatio > maxFontRatio(metrics) && !verticalColumnContinuation(metrics)) return false
  return sameTextAxis(metrics)
}

function verticalColumnContinuation(metrics: LinePairMetrics): boolean {
  return metrics.direction === 'vertical'
    && metrics.primaryGapFont <= 0.45
    && metrics.secondaryOverlap >= 0.72
    && metrics.centerShiftSpan <= 0.38
    && metrics.fontRatio <= 4.75
}

function strongLineContinuation(a: SpeechLine, b: SpeechLine, stats: LineGeometryStats): boolean {
  const metrics = linePairMetrics(a, b, stats.medianFontPx)
  return metrics ? verticalColumnContinuation(metrics) : false
}

function linePairMetrics(a: SpeechLine, b: SpeechLine, fallbackFontPx: number): LinePairMetrics | null {
  if (a.textDirection !== b.textDirection) return null
  const localFont = localFontPx(a, b, fallbackFontPx)
  const aFont = lineFontPx(a) || localFont
  const bFont = lineFontPx(b) || localFont
  const fontRatio = Math.max(aFont, bFont) / Math.max(1, Math.min(aFont, bFont))
  const angleSin = Math.abs(Math.sin((a.rotationDeg - b.rotationDeg) * Math.PI / 180))

  // Measure gap/overlap/shift in the text's own rotated frame.  On axis-aligned
  // bboxes a tilted line inflates to `w·sinθ + h·cosθ`, which shrinks the
  // gap/font ratio and falsely merges distant tilted blocks (e.g. stacked phone
  // chat bubbles).  The oriented frame uses true extents, so a real vertical gap
  // stays large and the merge guards reject it.
  const frame = orientedPairFrame(a.oriented, b.oriented, a.textDirection)
  const maxSecondarySpan = Math.max(1, frame.maxSecondarySpan)
  return {
    direction: a.textDirection,
    localFontPx: localFont,
    primaryGapPx: frame.primaryGapPx,
    primaryGapFont: frame.primaryGapPx / localFont,
    secondaryOverlap: frame.secondaryOverlap,
    centerShiftPx: frame.centerShiftPx,
    centerShiftFont: frame.centerShiftPx / localFont,
    centerShiftSpan: frame.centerShiftPx / maxSecondarySpan,
    fontRatio,
    angleSin,
    maxSecondarySpan,
  }
}

function textRegionCost(metrics: LinePairMetrics): number {
  const gapCost = metrics.primaryGapFont
  const overlapCost = 1 - clamp(metrics.secondaryOverlap, 0, 1)
  const centerCost = Math.min(metrics.centerShiftFont / 1.5, metrics.centerShiftSpan / 0.45)
  const fontCost = Math.log2(metrics.fontRatio)
  const angleCost = metrics.angleSin / 0.34
  return gapCost + overlapCost * 0.75 + centerCost * 0.65 + fontCost * 0.45 + angleCost * 0.75
}

function sameTextAxis(metrics: LinePairMetrics): boolean {
  const contained = metrics.secondaryOverlap >= 0.45
  const centered = metrics.centerShiftSpan <= 0.45 || metrics.centerShiftFont <= 1.35
  const notAcrossColumns = metrics.secondaryOverlap >= 0.25 || metrics.centerShiftSpan <= 0.58 || metrics.centerShiftPx <= metrics.localFontPx * 1.7
  return (contained || centered) && notAcrossColumns
}

function maxPrimaryGapFont(metrics: LinePairMetrics): number {
  const strongAxis = metrics.secondaryOverlap >= 0.55 || metrics.centerShiftSpan <= 0.32 || metrics.centerShiftFont <= 1.15
  if (metrics.fontRatio <= 1.35) return strongAxis ? 1.75 : 1.35
  if (metrics.fontRatio <= 2.1 && strongAxis) return 1.55
  return strongAxis ? 1.15 : 0.85
}

function maxFontRatio(metrics: LinePairMetrics): number {
  const veryClose = metrics.primaryGapFont <= 0.45
  const strongAxis = metrics.secondaryOverlap >= 0.60 || metrics.centerShiftSpan <= 0.30 || metrics.centerShiftFont <= 1.0
  if (veryClose && strongAxis) return 3.0
  if (strongAxis) return 2.25
  return 1.65
}

function localFontPx(a: SpeechLine, b: SpeechLine, fallback: number): number {
  return median([lineFontPx(a), lineFontPx(b)].filter(value => value > 0)) || fallback
}

function hardNegativeLineConflict(a: SpeechLine, b: SpeechLine, stats: LineGeometryStats): boolean {
  const metrics = linePairMetrics(a, b, stats.medianFontPx)
  if (!metrics) return true
  const fontPx = metrics.localFontPx
  const intersection = intersectionArea(a.bbox, b.bbox)
  const noiseArea = fontPx * fontPx * OCR_BOX_NOISE_FRACTION
  const meaningfulOverlap = intersection > noiseArea && intersection / Math.max(1, Math.min(bboxArea(a.bbox), bboxArea(b.bbox))) > OCR_BOX_NOISE_FRACTION
  if (meaningfulOverlap && !relatedTextForGrouping(a.text, b.text)) return true

  if (metrics.primaryGapFont > 1.8) return false
  if (metrics.secondaryOverlap >= 0.55) return false
  const expectedShift = metrics.maxSecondarySpan * (1 - metrics.secondaryOverlap)
  return metrics.centerShiftPx > expectedShift + fontPx * Math.max(1, stats.edgeCostScale)
}

function relatedTextForGrouping(a: string, b: string): boolean {
  const left = normalizeForGrouping(a)
  const right = normalizeForGrouping(b)
  if (!left || !right) return false
  return left.includes(right) || right.includes(left)
}

function samePunctuationIntent(a: string, b: string): boolean {
  const left = punctuationIntent(a)
  const right = punctuationIntent(b)
  return left !== null && left === right
}

function punctuationIntent(text: string): 'ellipsis' | 'emphasis' | 'dash' | null {
  const compact = text.replace(/\s+/gu, '')
  if (!compact || !/^[\p{P}\p{S}]+$/u.test(compact)) return null
  if (/…|⋯|\.{2,}|。{2,}/u.test(compact)) return 'ellipsis'
  if (/^[!?！？]+$/u.test(compact)) return 'emphasis'
  if (/^[—~〜-]+$/u.test(compact)) return 'dash'
  return null
}

function normalizeForGrouping(text: string): string {
  return text.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
}

function centerX(bbox: BBox): number {
  return (bbox[0] + bbox[2]) / 2
}

function centerY(bbox: BBox): number {
  return (bbox[1] + bbox[3]) / 2
}

function mergeSpeechLineGroup(group: readonly SpeechLine[]): TextBlock {
  const ordered = [...group].sort(compareSpeechLinesForReading)
  if (ordered.length === 1) return speechLineToTextBlock(ordered[0]!)
  const bbox = unionBBoxes(ordered.map(line => line.bbox))
  const direction = majorityDirection(ordered)
  return {
    bbox,
    polygon: bboxToPolygon(bbox),
    text: ordered.map(line => line.text).filter(Boolean).join('\n'),
    rotationDeg: maxAbsRotation(ordered),
    textDirection: direction,
    confidence: Math.max(...ordered.map(line => line.confidence)),
    lines: ordered.map(line => line.line),
    words: ordered.flatMap(line => line.words),
  }
}

function speechLineToTextBlock(line: SpeechLine): TextBlock {
  return {
    bbox: line.bbox,
    polygon: bboxToPolygon(line.bbox),
    text: line.text,
    rotationDeg: line.rotationDeg,
    textDirection: line.textDirection,
    confidence: line.confidence,
    lines: [line.line],
    words: line.words,
  }
}

function compareSpeechLinesForReading(a: SpeechLine, b: SpeechLine): number {
  const vertical = a.textDirection === 'vertical' || b.textDirection === 'vertical'
  return vertical
    ? b.bbox[0] - a.bbox[0] || a.bbox[1] - b.bbox[1]
    : a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]
}

function compareBlocksForReading(a: TextBlock, b: TextBlock): number {
  const vertical = a.textDirection === 'vertical' || b.textDirection === 'vertical'
  return vertical
    ? b.bbox[0] - a.bbox[0] || a.bbox[1] - b.bbox[1]
    : a.bbox[1] - b.bbox[1] || a.bbox[0] - b.bbox[0]
}

function majorityDirection(blocks: readonly { readonly textDirection: TextDirection }[]): TextDirection {
  return blocks.filter(block => block.textDirection === 'vertical').length * 2 >= blocks.length ? 'vertical' : 'horizontal'
}

function normalizedCharCount(text: string): number {
  return [...text].filter(ch => !/\s/u.test(ch)).length
}

function intersectionArea(a: BBox, b: BBox): number {
  const ix1 = Math.max(a[0], b[0])
  const iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2])
  const iy2 = Math.min(a[3], b[3])
  if (ix1 >= ix2 || iy1 >= iy2) return 0
  return (ix2 - ix1) * (iy2 - iy1)
}

function unionBBoxes(boxes: readonly BBox[]): BBox {
  return [Math.min(...boxes.map(box => box[0])), Math.min(...boxes.map(box => box[1])), Math.max(...boxes.map(box => box[2])), Math.max(...boxes.map(box => box[3]))]
}

function bboxToPolygon(bbox: BBox): Polygon {
  return [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
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

function maxAbsRotation(blocks: readonly { readonly rotationDeg: number }[]): number {
  return blocks.reduce((best, block) => Math.abs(block.rotationDeg) > Math.abs(best) ? block.rotationDeg : best, 0)
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}

function robustScale(values: readonly number[], med = median(values)): number {
  if (!values.length) return 0
  const mad = median(values.map(value => Math.abs(value - med)))
  return mad * 1.4826
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export async function translatePreparedText(args: {
  readonly text: PreparedTextResult
  readonly translator: Translator
  readonly sourceLanguage: string | null
  readonly targetLanguage: string
  readonly signal?: AbortSignal
}): Promise<PreparedTranslationResult> {
  const translatable = args.text.translationUnits.filter(unit => unit.kind !== 'skip' && unit.sourceText.trim())
  const translated = translatable.length
    ? await args.translator.translateUnits({
      units: translatable,
      sourceLang: args.sourceLanguage,
      targetLang: args.targetLanguage,
      signal: args.signal,
    })
    : []
  const byUnitId = new Map(translated.map(unit => [unit.unitId, unit]))
  const translations = args.text.translationUnits.map(unit => {
    const translated = byUnitId.get(unit.id)
    return translated ? postProcessTranslation(translated, args.targetLanguage) : skippedTranslation(unit)
  })

  return {
    ...args.text,
    translations,
  }
}

function postProcessTranslation(unit: TranslatedUnit, targetLanguage: string): TranslatedUnit {
  const targetText = normalizeTargetPunctuation(unit.targetText, targetLanguage)
  return {
    ...unit,
    kind: targetText.trim() ? unit.kind : 'skip',
    targetText,
  }
}

function normalizeTargetPunctuation(text: string, targetLanguage: string): string {
  if (preserveCjkEllipsis(targetLanguage)) return text.trim()
  const normalized = text
    .replace(/(?:\.{3,}|…{2,}|⋯{2,}|(?:\.[ \t]*){3,})/gu, '…')
    .replace(/(?:…[ \t]*){2,}/gu, '…')
    .replace(/[ \t]+/gu, ' ')
    .trim()
  return dedupeStandaloneEllipsisLines(normalized)
}

function dedupeStandaloneEllipsisLines(text: string): string {
  const lines = text.split(/\r?\n/u).map(line => line.trim()).filter(Boolean)
  if (lines.length <= 1) return text
  const result: string[] = []
  let previousWasEllipsis = false
  for (const line of lines) {
    if (/^…+$/u.test(line)) {
      if (previousWasEllipsis) continue
      previousWasEllipsis = true
      result.push('…')
      continue
    }
    previousWasEllipsis = false
    result.push(line)
  }
  return result.join('\n')
}

function preserveCjkEllipsis(targetLanguage: string): boolean {
  return /^(zh|ja)\b/iu.test(targetLanguage.trim())
}

function skippedTranslation(unit: TranslationUnit): TranslatedUnit {
  return {
    unitId: unit.id,
    pageIndex: unit.pageIndex,
    kind: 'skip',
    role: unit.role,
    sourceText: unit.sourceText,
    targetText: '',
  }
}
