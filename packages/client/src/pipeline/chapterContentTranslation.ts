import type { ChapterContentLayout, ChapterOverlay } from '../domain/chapterContent'
import type { BBox, Polygon } from '../domain/geometry'
import type { ImagePixels } from '../domain/image'
import type { TextPlacement } from '../domain/planning'
import type { TextRegion } from '../domain/regions'
import type { RecognizedTextPage } from '../domain/text'
import type { TranslationUnit, TranslatedUnit } from '../domain/translation'
import type { Translator } from '../translators/translator'
import type { SafeMarginsDebug } from '../render/backgroundFit'
import { estimateSafeMargins } from '../render/backgroundFit'
import { textFitRect } from '../render/fitGeometry'
import type { ChapterOcrChunk } from './chapterContent'
import { mapChunkRecognitionToChapter } from './chapterContent'
import { buildOverlayPlacements } from './composeOverlay'
import { textFromRecognition, translatePreparedText } from './translatePreparedPage'

export interface ChapterContentOverlay extends ChapterOverlay {
  readonly placementMargins: readonly SafeMarginsDebug[]
}

export interface ChapterOverlaySlice {
  readonly placements: readonly TextPlacement[]
  readonly translations: readonly TranslatedUnit[]
  readonly placementMargins: readonly SafeMarginsDebug[]
}

export interface OverlayPlacementItem {
  readonly placement: TextPlacement
  readonly margin: SafeMarginsDebug
}

export interface TranslateChapterContentChunkArgs {
  readonly recognized: RecognizedTextPage
  readonly chunk: ChapterOcrChunk
  readonly layout: ChapterContentLayout
  readonly image: ImagePixels
  readonly regions?: readonly TextRegion[] | null
  readonly translator: () => Translator
  readonly sourceLanguage: string | null
  readonly targetLanguage: string
  readonly signal?: AbortSignal
}

const CHUNK_CROP_TOUCH_GUARD_PX = 48

export function emptyChapterContentOverlay(layout: ChapterContentLayout): ChapterContentOverlay {
  return { contentSize: layout.contentSize, placements: [], translations: [], placementMargins: [] }
}

export async function translateChapterContentChunk(args: TranslateChapterContentChunkArgs): Promise<ChapterOverlaySlice> {
  const mapped = mapChunkRecognitionToChapter(args.recognized, args.chunk)
  const clean = removeReaderNoiseBlocks(mapped, args.layout)
  const regions = mapChunkRegionsToChapter(args.regions ?? null, args.chunk, args.layout.contentSize)
  const text = textFromRecognition({ pageIndex: args.chunk.index, recognized: clean, regions })

  const sourceByUnitId = new Map(text.textUnits.map(unit => [unit.id, unit.sourceText]))
  const placements = buildOverlayPlacements({
    recognized: text.recognized,
    textUnits: text.textUnits,
    regions,
  })
    .map(placement => ({ ...placement, pageSize: [args.layout.contentSize.width, args.layout.contentSize.height] as const }))
    .filter(placement => isCompletePlacement(placement, args.chunk, args.layout, sourceByUnitId))

  if (!placements.length) return emptyOverlaySlice()

  const translationInput = translationInputFromPlacements(placements, text.translationUnits)
  const renderPlacements = placements.map(placement => ({ ...placement, sourceUnitIds: renderSourceUnitIds(placement) }))
  const renderText = {
    ...text,
    textUnits: [],
    translationUnits: translationInput,
  }

  if (!hasTranslatableUnit(renderText.translationUnits)) return emptyOverlaySlice()

  const translated = await translatePreparedText({
    text: renderText,
    translator: args.translator(),
    sourceLanguage: args.sourceLanguage,
    targetLanguage: args.targetLanguage,
    signal: args.signal,
  })

  const margins = estimateChunkPlacementMargins(args.image, args.chunk, renderPlacements)
  return projectChunkPlacementsToPages(renderPlacements, margins, args.layout, translated.translations)
}

function translationInputFromPlacements(
  placements: readonly TextPlacement[],
  units: readonly TranslationUnit[],
): readonly TranslationUnit[] {
  const byId = new Map(units.map(unit => [unit.id, unit]))
  return placements.map(placement => {
    const sourceUnits = placement.sourceUnitIds.map(id => byId.get(id)).filter((unit): unit is TranslationUnit => !!unit)
    const translatableUnits = sourceUnits.filter(unit => unit.kind !== 'skip' && unit.sourceText.trim())
    const sourceText = translatableUnits.map(unit => unit.sourceText).filter(Boolean).join('\n')
    const skipped = translatableUnits.length === 0
    return {
      id: placementUnitId(placement),
      pageIndex: placement.pageIndex,
      blockIds: translatableUnits.flatMap(unit => unit.blockIds),
      sourceText,
      kind: skipped || !sourceText.trim() ? 'skip' : placement.role === 'sfx' ? 'sfx' : 'dialogue',
      role: placement.role,
    }
  })
}

function placementUnitId(placement: TextPlacement): string {
  return `${placement.id}-unit`
}

function renderSourceUnitIds(placement: TextPlacement): readonly string[] {
  const syntheticId = placementUnitId(placement)
  return placement.sourceUnitIds.length > 1 ? [syntheticId, ...placement.sourceUnitIds] : [syntheticId]
}

function mapChunkRegionsToChapter(
  regions: readonly TextRegion[] | null,
  chunk: ChapterOcrChunk,
  contentSize: ChapterContentLayout['contentSize'],
): readonly TextRegion[] | null {
  if (!regions?.length) return null
  return regions.map(region => ({
    ...region,
    bbox: clipBBox(translateBBox(region.bbox, chunk.contentRect.x, chunk.contentRect.y), contentSize),
  }))
}

function clipBBox(bbox: BBox, contentSize: ChapterContentLayout['contentSize']): BBox {
  return [
    Math.max(0, bbox[0]),
    Math.max(0, bbox[1]),
    Math.min(contentSize.width, bbox[2]),
    Math.min(contentSize.height, bbox[3]),
  ]
}

export function mergeChapterContentOverlay(
  existing: ChapterContentOverlay,
  incoming: ChapterOverlaySlice,
): ChapterContentOverlay {
  const translations = mergeTranslations(existing.translations, incoming.translations)
  const byUnitId = new Map(translations.map(unit => [unit.unitId, unit]))
  const items = [
    ...existing.placements.map((placement, index) => ({ placement, margin: existing.placementMargins[index]! })),
    ...incoming.placements.map((placement, index) => ({ placement, margin: incoming.placementMargins[index]! })),
  ]
    .sort((a, b) => placementScore(b.placement, byUnitId) - placementScore(a.placement, byUnitId))
    .reduce<OverlayPlacementItem[]>((kept, item) => {
      if (!kept.some(candidate => duplicatePlacement(candidate.placement, item.placement, byUnitId))) kept.push(item)
      return kept
    }, [])
    .sort((a, b) => a.placement.bbox[1] - b.placement.bbox[1] || a.placement.bbox[0] - b.placement.bbox[0])
  return { ...existing, translations, placements: items.map(item => item.placement), placementMargins: items.map(item => item.margin) }
}

function emptyOverlaySlice(): ChapterOverlaySlice {
  return { placements: [], translations: [], placementMargins: [] }
}

function hasTranslatableUnit(units: readonly { readonly kind: string; readonly sourceText: string }[]): boolean {
  return units.some(unit => unit.kind !== 'skip' && unit.sourceText.trim())
}

function removeReaderNoiseBlocks(recognized: RecognizedTextPage, layout: ChapterContentLayout): RecognizedTextPage {
  return { ...recognized, blocks: recognized.blocks.filter(block => !isReaderNoiseText(block.text, block.bbox, noiseFrameForBBox(layout, block.bbox))) }
}

function isCompletePlacement(
  placement: TextPlacement,
  chunk: ChapterOcrChunk,
  layout: ChapterContentLayout,
  sourceByUnitId: ReadonlyMap<string, string>,
): boolean {
  if (!isChunkOwner(placement.bbox, chunk)) return false
  if (touchesChunkCropEdge(placement.bbox, chunk, layout.contentSize.height)) return false
  if (isReaderNoiseText(placementSource(placement, sourceByUnitId), placement.bbox, noiseFrameForBBox(layout, placement.bbox))) return false
  return true
}

function estimateChunkPlacementMargins(
  image: ImagePixels,
  chunk: ChapterOcrChunk,
  placements: readonly TextPlacement[],
): readonly SafeMarginsDebug[] {
  const pageSize: readonly [number, number] = [image.width, image.height]
  const chunkPlacements = placements.map(placement => translatePlacement(placement, -chunk.contentRect.x, -chunk.contentRect.y, pageSize))
  return chunkPlacements.map((placement, index) => {
    const baseRect = textFitRect(placement)
    const obstacles = chunkPlacements
      .filter((_, i) => i !== index)
      .flatMap(placementBBoxes)
    return translateMargin(estimateSafeMargins({ image, placement, baseRect, obstacles, pageSize }), chunk.contentRect.x, chunk.contentRect.y)
  })
}

function placementBBoxes(placement: TextPlacement): readonly BBox[] {
  return placement.textBoxes.length ? placement.textBoxes : [placement.bbox]
}

function translatePlacement(
  placement: TextPlacement,
  dx: number,
  dy: number,
  pageSize: readonly [number, number],
): TextPlacement {
  return {
    ...placement,
    pageSize,
    bbox: translateBBox(placement.bbox, dx, dy),
    drawable: translatePolygon(placement.drawable, dx, dy),
    textBoxes: placement.textBoxes.map(box => translateBBox(box, dx, dy)),
  }
}

function translateMargin(margin: SafeMarginsDebug, dx: number, dy: number): SafeMarginsDebug {
  return {
    ...margin,
    safeBounds: translateBBox(margin.safeBounds, dx, dy),
    componentBBox: margin.componentBBox ? translateBBox(margin.componentBBox, dx, dy) : null,
    shape: margin.shape ? {
      ...margin.shape,
      bounds: translateBBox(margin.shape.bounds, dx, dy),
      spans: margin.shape.spans.map(span => ({ y: span.y + dy, x1: span.x1 + dx, x2: span.x2 + dx })),
    } : null,
  }
}

function projectChunkPlacementsToPages(
  placements: readonly TextPlacement[],
  margins: readonly SafeMarginsDebug[],
  layout: ChapterContentLayout,
  translations: readonly TranslatedUnit[],
): ChapterOverlaySlice {
  const out: Array<{ placement: TextPlacement; margin: SafeMarginsDebug }> = []
  for (let i = 0; i < placements.length; i += 1) {
    const placement = placements[i]!
    const page = ownerPageForPlacement(placement, layout)
    if (!page) continue
    const projected = projectPlacementToPage(placement, page)
    if (!projected) continue
    out.push({ placement: projected, margin: projectMarginToPage(margins[i]!, page, projected.bbox) })
  }
  return {
    placements: out.map(item => item.placement),
    translations,
    placementMargins: out.map(item => item.margin),
  }
}

function ownerPageForPlacement(placement: TextPlacement, layout: ChapterContentLayout): ChapterContentLayout['pages'][number] | null {
  let best: ChapterContentLayout['pages'][number] | null = null
  let bestArea = 0
  for (const page of layout.pages) {
    const area = intersectionArea(placement.bbox, rectBBox(page.contentRect))
    if (area > bestArea) {
      best = page
      bestArea = area
    }
  }
  if (best) return best
  const cy = (placement.bbox[1] + placement.bbox[3]) / 2
  return layout.pages.find(page => page.contentRect.y <= cy && cy <= page.contentRect.y + page.contentRect.height) ?? null
}

function projectPlacementToPage(placement: TextPlacement, page: ChapterContentLayout['pages'][number]): TextPlacement | null {
  const pageSize = [page.sourceSize.width, page.sourceSize.height] as const
  const bbox = clipBBoxToPage(projectBBoxToPage(placement.bbox, page), pageSize)
  if (!bbox) return null
  return {
    ...placement,
    pageIndex: page.pageIndex,
    pageSize,
    fontHint: projectFontHintToPage(placement.fontHint, page),
    bbox,
    drawable: projectPolygonToPage(placement.drawable, page, pageSize),
    textBoxes: placement.textBoxes
      .map(box => clipBBoxToPage(projectBBoxToPage(box, page), pageSize))
      .filter((box): box is BBox => box !== null),
  }
}

function projectFontHintToPage(fontHint: TextPlacement['fontHint'], page: ChapterContentLayout['pages'][number]): TextPlacement['fontHint'] {
  if (!fontHint?.sourceFontPx) return fontHint
  const scaleY = page.sourceSize.height / Math.max(1, page.contentRect.height)
  return { ...fontHint, sourceFontPx: fontHint.sourceFontPx * scaleY }
}

function projectMarginToPage(
  margin: SafeMarginsDebug,
  page: ChapterContentLayout['pages'][number],
  fallbackBounds: BBox,
): SafeMarginsDebug {
  const pageSize = [page.sourceSize.width, page.sourceSize.height] as const
  const scaleX = page.sourceSize.width / Math.max(1, page.contentRect.width)
  const scaleY = page.sourceSize.height / Math.max(1, page.contentRect.height)
  return {
    ...margin,
    margins: {
      top: margin.margins.top * scaleY,
      right: margin.margins.right * scaleX,
      bottom: margin.margins.bottom * scaleY,
      left: margin.margins.left * scaleX,
    },
    safeBounds: clipBBoxToPage(projectBBoxToPage(margin.safeBounds, page), pageSize) ?? fallbackBounds,
    componentBBox: margin.componentBBox ? clipBBoxToPage(projectBBoxToPage(margin.componentBBox, page), pageSize) : null,
    shape: margin.shape ? {
      ...margin.shape,
      bounds: clipBBoxToPage(projectBBoxToPage(margin.shape.bounds, page), pageSize) ?? fallbackBounds,
      spans: margin.shape.spans.map(span => ({
        y: (span.y - page.contentRect.y) * scaleY,
        x1: clamp((span.x1 - page.contentRect.x) * scaleX, 0, page.sourceSize.width),
        x2: clamp((span.x2 - page.contentRect.x) * scaleX, 0, page.sourceSize.width),
      })),
    } : null,
  }
}

function projectBBoxToPage(bbox: BBox, page: ChapterContentLayout['pages'][number]): BBox {
  const scaleX = page.sourceSize.width / Math.max(1, page.contentRect.width)
  const scaleY = page.sourceSize.height / Math.max(1, page.contentRect.height)
  return [
    (bbox[0] - page.contentRect.x) * scaleX,
    (bbox[1] - page.contentRect.y) * scaleY,
    (bbox[2] - page.contentRect.x) * scaleX,
    (bbox[3] - page.contentRect.y) * scaleY,
  ]
}

function projectPolygonToPage(polygon: Polygon, page: ChapterContentLayout['pages'][number], pageSize: readonly [number, number]): Polygon {
  const scaleX = page.sourceSize.width / Math.max(1, page.contentRect.width)
  const scaleY = page.sourceSize.height / Math.max(1, page.contentRect.height)
  return polygon.map(point => [
    clamp((point[0] - page.contentRect.x) * scaleX, 0, pageSize[0]),
    clamp((point[1] - page.contentRect.y) * scaleY, 0, pageSize[1]),
  ])
}

function rectBBox(rect: ChapterContentLayout['pages'][number]['contentRect']): BBox {
  return [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]
}

function intersectionArea(a: BBox, b: BBox): number {
  const x1 = Math.max(a[0], b[0])
  const y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2])
  const y2 = Math.min(a[3], b[3])
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
}

function clipBBoxToPage(bbox: BBox, pageSize: readonly [number, number]): BBox | null {
  const x1 = clamp(bbox[0], 0, pageSize[0])
  const y1 = clamp(bbox[1], 0, pageSize[1])
  const x2 = clamp(bbox[2], 0, pageSize[0])
  const y2 = clamp(bbox[3], 0, pageSize[1])
  return x1 < x2 && y1 < y2 ? [x1, y1, x2, y2] : null
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}

function translateBBox(bbox: BBox, dx: number, dy: number): BBox {
  return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]
}

function translatePolygon(polygon: Polygon, dx: number, dy: number): Polygon {
  return polygon.map(point => [point[0] + dx, point[1] + dy])
}

function mergeTranslations(
  first: readonly TranslatedUnit[],
  second: readonly TranslatedUnit[],
): readonly TranslatedUnit[] {
  const out = new Map<string, TranslatedUnit>()
  for (const unit of first) out.set(unit.unitId, unit)
  for (const unit of second) out.set(unit.unitId, unit)
  return [...out.values()]
}

function duplicatePlacement(
  a: TextPlacement,
  b: TextPlacement,
  translations: ReadonlyMap<string, TranslatedUnit>,
): boolean {
  if (a.pageIndex !== b.pageIndex) return false
  const sourceA = placementSource(a, translations)
  const sourceB = placementSource(b, translations)
  if (!relatedText(sourceA, sourceB)) return false
  return bboxOverlapRatio(a.bbox, b.bbox) >= 0.55 || centerDistanceRatio(a.bbox, b.bbox) <= 0.30
}

function placementScore(
  placement: TextPlacement,
  translations: ReadonlyMap<string, TranslatedUnit>,
): number {
  return normalizeText(placementSource(placement, translations)).length * 10
    + placement.textBoxes.length * 3
    + bboxArea(placement.bbox) / 10000
}

function placementSource(
  placement: TextPlacement,
  sourceByUnitId: ReadonlyMap<string, string | { readonly sourceText: string }>,
): string {
  return placement.sourceUnitIds.map(id => {
    const source = sourceByUnitId.get(id)
    return typeof source === 'string' ? source : source?.sourceText ?? ''
  }).join('\n')
}

interface NoiseFrame {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

function isReaderNoiseText(text: string, bbox: BBox, frame: NoiseFrame): boolean {
  const source = normalizeText(text)
  if (!source) return true
  if (source === '巅峰' || source === '武炼巅峰' || source === '武煉巔峰') return true
  if (isCreditFragment(source) && isSmallEdgeText(source, bbox, frame)) return true
  if (/^[0-9]{1,4}$/u.test(source) && bboxArea(bbox) < 6000) return true
  if (/^[a-z0-9]$/iu.test(source) && bboxArea(bbox) < 2000) return true
  return false
}

function noiseFrameForBBox(layout: ChapterContentLayout, bbox: BBox): NoiseFrame {
  const cy = (bbox[1] + bbox[3]) / 2
  const page = layout.pages.find(page => page.contentRect.y <= cy && cy <= page.contentRect.y + page.contentRect.height)
  return page?.contentRect ?? { x: 0, y: 0, width: layout.contentSize.width, height: layout.contentSize.height }
}

function isCreditFragment(source: string): boolean {
  return source === '腾讯'
    || source === '騰訊'
    || source === '腾'
    || source === '訊'
    || source === '讯'
    || /^(?:腾讯|騰訊)(?:动|動|动漫|動漫|漫|漫画|漫畫)?$/u.test(source)
    || source === 'tencent'
    || source.includes('tencentcomics')
    || source.includes('tencentanime')
    || source === '包子'
    || source.includes('包子漫')
    || source.includes('baozimh')
}

function isSmallEdgeText(source: string, bbox: BBox, frame: NoiseFrame): boolean {
  const width = Math.max(1, frame.width)
  const height = Math.max(1, frame.height)
  const boxWidth = bboxWidth(bbox)
  const boxHeight = bboxHeight(bbox)
  const small = boxWidth <= width * 0.30
    && boxHeight <= Math.min(width * 0.09, height * 0.10)
    && bboxArea(bbox) <= width * height * 0.015
  const localX1 = bbox[0] - frame.x
  const localX2 = bbox[2] - frame.x
  const localY1 = bbox[1] - frame.y
  const localY2 = bbox[3] - frame.y
  const nearSideEdge = localX1 <= width * 0.18 || localX2 >= width * 0.82
  const nearVerticalEdge = localY1 <= height * 0.12 || localY2 >= height * 0.88
  if (source === '包子') return small && nearSideEdge && nearVerticalEdge
  return small && (nearSideEdge || nearVerticalEdge)
}

function isChunkOwner(bbox: BBox, chunk: ChapterOcrChunk): boolean {
  const cy = (bbox[1] + bbox[3]) / 2
  const coreTop = chunk.coreRect.y
  const coreBottom = chunk.coreRect.y + chunk.coreRect.height
  return cy >= coreTop && cy <= coreBottom
}

function touchesChunkCropEdge(
  bbox: BBox,
  chunk: ChapterOcrChunk,
  contentHeight: number,
): boolean {
  const topEdge = chunk.contentRect.y
  const bottomEdge = chunk.contentRect.y + chunk.contentRect.height
  const hasPreviousChunk = topEdge > 0
  const hasNextChunk = bottomEdge < contentHeight
  if (hasPreviousChunk && bbox[1] < topEdge + CHUNK_CROP_TOUCH_GUARD_PX) return true
  if (hasNextChunk && bbox[3] > bottomEdge - CHUNK_CROP_TOUCH_GUARD_PX) return true
  return false
}

function relatedText(a: string, b: string): boolean {
  const left = normalizeText(a)
  const right = normalizeText(b)
  if (!left || !right) return false
  return left.includes(right) || right.includes(left) || longestCommonSubstringLength(left, right) / Math.min(left.length, right.length) >= 0.45
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
}

function longestCommonSubstringLength(a: string, b: string): number {
  let previous = new Array(b.length + 1).fill(0) as number[]
  let best = 0
  for (let i = 1; i <= a.length; i++) {
    const current = new Array(b.length + 1).fill(0) as number[]
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] !== b[j - 1]) continue
      const length = previous[j - 1]! + 1
      current[j] = length
      if (length > best) best = length
    }
    previous = current
  }
  return best
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
