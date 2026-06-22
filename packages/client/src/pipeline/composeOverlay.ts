/**
 * Compose placements + translations into one or more PageOverlay objects.
 *
 * This is where the OCR/ONNX fork merges — ONNX regions are preferred
 * when available, falling back to OCR-only placement.
 *
 * Background expansion (SafeMargins) is pre-computed here from the image
 * so the renderer never needs raw pixel access.
 */

import type { TextRegion } from '../domain/regions'
import type { PageOverlay } from '../domain/overlay'
import type { PreparedPageHandle } from '../domain/prepared'
import type { TranslatedUnit } from '../domain/translation'
import type { RecognizedTextPage, TextUnit } from '../domain/text'
import type { TextPlacement } from '../domain/planning'
import type { ImagePixels } from '../domain/image'
import { clipBBox, type BBox, type Polygon } from '../domain/geometry'
import type { SafeMarginsDebug } from '../render/backgroundFit'
import { estimateSafeMargins } from '../render/backgroundFit'
import { textFitRect } from '../render/fitGeometry'
import type { TextRoleContext } from './textRole'
import {
  textPlacementsFromRecognition,
  layoutPlacementsFromRegions,
} from './textPlacements'

export interface ComposeOverlayArgs {
  readonly handle: PreparedPageHandle
  readonly recognized: RecognizedTextPage
  readonly textUnits: readonly TextUnit[]
  readonly translations: readonly TranslatedUnit[]
  readonly regions: readonly TextRegion[] | null
  readonly placements?: readonly TextPlacement[]
  readonly placementMargins?: readonly SafeMarginsDebug[]
  /** Optional fallback for legacy callers that still compute margins on main. */
  readonly sourceImage?: ImagePixels
}

export function buildOverlayPlacements(args: {
  readonly recognized: RecognizedTextPage
  readonly textUnits: readonly TextUnit[]
  readonly regions: readonly TextRegion[] | null
  readonly roleContext?: TextRoleContext
}): readonly TextPlacement[] {
  return args.regions?.length
    ? layoutPlacementsFromRegions(
        args.recognized,
        args.textUnits,
        args.regions,
        args.roleContext,
      )
    : textPlacementsFromRecognition(args.recognized, args.textUnits, args.roleContext)
}

export function composeAndProjectOverlays(args: ComposeOverlayArgs): readonly PageOverlay[] {
  const placements = args.placements ?? buildOverlayPlacements(args)
  const margins = args.placementMargins ?? estimateMarginsOnMain(args.sourceImage, placements)

  if (margins.length !== placements.length) {
    throw new Error('placement margin count mismatch')
  }

  return projectPreparedPlacements({
    handle: args.handle,
    placements,
    translations: args.translations,
    margins,
  })
}

function estimateMarginsOnMain(
  sourceImage: ImagePixels | undefined,
  placements: readonly TextPlacement[],
): readonly SafeMarginsDebug[] {
  if (!sourceImage) throw new Error('sourceImage or placementMargins is required')
  const pageSize: readonly [number, number] = [sourceImage.width, sourceImage.height]
  return placements.map((placement, index) => {
    const baseRect = textFitRect(placement)
    const others = placements
      .filter((_, i) => i !== index)
      .flatMap(placementBBoxes)
    return estimateSafeMargins({
      image: sourceImage,
      placement,
      baseRect,
      obstacles: others,
      pageSize,
    })
  })
}

function placementBBoxes(placement: TextPlacement) {
  return placement.textBoxes.length ? [...placement.textBoxes] : [placement.bbox]
}

function projectPreparedPlacements(args: {
  readonly handle: PreparedPageHandle
  readonly placements: readonly TextPlacement[]
  readonly translations: readonly TranslatedUnit[]
  readonly margins: readonly SafeMarginsDebug[]
}): PageOverlay[] {
  const { handle, placements, translations, margins } = args
  if (!handle.projections.length) {
    throw new Error(
      `PreparedPageHandle ${handle.preparedPageId} has no projections`,
    )
  }

  const skipUnitIds = new Set(translations.filter(unit => unit.kind === 'skip').map(unit => unit.unitId))
  const translationsByUnitId = new Map(translations.map(unit => [unit.unitId, unit]))

  // Group placements by target source page.
  const bySourcePage = new Map<number, { placement: TextPlacement; margin: SafeMarginsDebug; mutedUnitIds: readonly string[] }[]>()

  for (let i = 0; i < placements.length; i++) {
    const placement = placements[i]!
    if (placement.sourceUnitIds.length && placement.sourceUnitIds.every(id => skipUnitIds.has(id))) continue
    const projections = projectionsForPlacement(handle, placement)
    const owner = ownerProjection(placement, projections)
    const margin = margins[i]!

    for (const projection of projections) {
      const sourceIndex = projection.sourcePageIndex
      const projected = projectPlacement(placement, projection)
      if (!projected) continue
      if (isEdgeCreditFragment(projected, translationsByUnitId)) continue
      const projectedMargin = projectMargin(margin, projection, projected.bbox)
      const mutedUnitIds = projections.length > 1 && projection !== owner ? placement.sourceUnitIds : []

      const list = bySourcePage.get(sourceIndex)
      if (list) {
        list.push({ placement: projected, margin: projectedMargin, mutedUnitIds })
      } else {
        bySourcePage.set(sourceIndex, [{ placement: projected, margin: projectedMargin, mutedUnitIds }])
      }
    }
  }

  return [...bySourcePage.entries()].map(([pageIndex, items]) => ({
    pageIndex,
    pageSize: pageSizeForSource(handle, pageIndex),
    placements: items.map(i => i.placement),
    translations: translationsForItems(translations, items),
    placementMargins: items.map(i => i.margin),
  }))
}

function isEdgeCreditFragment(
  placement: TextPlacement,
  translationsByUnitId: ReadonlyMap<string, TranslatedUnit>,
): boolean {
  const sourceText = placement.sourceUnitIds
    .map(id => translationsByUnitId.get(id)?.sourceText ?? '')
    .join('')
  const text = sourceText.toLowerCase().replace(/[\s\p{P}\p{S}]+/gu, '')
  if (!isCreditFragmentText(text)) return false

  const [width, height] = placement.pageSize
  const [x1, y1, x2, y2] = placement.bbox
  const boxWidth = x2 - x1
  const boxHeight = y2 - y1
  const small = boxWidth <= width * 0.30
    && boxHeight <= height * 0.10
    && boxWidth * boxHeight <= width * height * 0.015
  const nearEdge = x1 <= width * 0.18
    || x2 >= width * 0.82
    || y1 <= height * 0.12
    || y2 >= height * 0.88
  if (text === '包子') return small && (x1 <= width * 0.18 || x2 >= width * 0.82) && (y1 <= height * 0.12 || y2 >= height * 0.88)
  return small && nearEdge
}

function isCreditFragmentText(text: string): boolean {
  return text === '腾讯'
    || text === '騰訊'
    || text === '腾'
    || text === '訊'
    || text === '讯'
    || text === 'tencent'
    || /^(?:腾讯|騰訊)(?:动|動|动漫|動漫|漫|漫画|漫畫)?$/u.test(text)
    || text.includes('tencentcomics')
    || text.includes('tencentanime')
    || text === '包子'
    || text.includes('包子漫')
    || text.includes('baozimh')
}

function ownerProjection(
  placement: TextPlacement,
  projections: readonly PreparedPageHandle['projections'][number][],
): PreparedPageHandle['projections'][number] | null {
  let best: PreparedPageHandle['projections'][number] | null = null
  let bestArea = -1
  for (const projection of projections) {
    const rect = projection.preparedRect
    const area = intersectionArea(placement.bbox, [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height])
    if (area > bestArea) {
      best = projection
      bestArea = area
    }
  }
  return best
}

function translationsForItems(
  translations: readonly TranslatedUnit[],
  items: readonly { readonly mutedUnitIds: readonly string[] }[],
): readonly TranslatedUnit[] {
  const muted = new Set(items.flatMap(item => item.mutedUnitIds))
  if (!muted.size) return translations
  return translations.map(unit => muted.has(unit.unitId)
    ? { ...unit, kind: 'skip', targetText: '' }
    : unit)
}

/** Return all projections whose prepared rect overlaps with the placement bbox. */
function projectionsForPlacement(handle: PreparedPageHandle, placement: TextPlacement) {
  return handle.projections.filter(projection => {
    const rect = projection.preparedRect
    return bboxOverlap(placement.bbox, [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height])
  })
}

function bboxOverlap(a: BBox, b: BBox): boolean {
  return Math.max(a[0], b[0]) < Math.min(a[2], b[2]) && Math.max(a[1], b[1]) < Math.min(a[3], b[3])
}

function intersectionArea(a: BBox, b: BBox): number {
  const x1 = Math.max(a[0], b[0])
  const y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2])
  const y2 = Math.min(a[3], b[3])
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
}

function projectPlacement(placement: TextPlacement, projection: PreparedPageHandle['projections'][number]): TextPlacement | null {
  const dx = projection.sourceRect.x - projection.preparedRect.x
  const dy = projection.sourceRect.y - projection.preparedRect.y
  const pageSize = projectionPageSize(projection)
  const bbox = clipBBoxToPage(translateBBox(placement.bbox, dx, dy), pageSize)
  if (!bbox) return null
  return {
    ...placement,
    pageIndex: projection.sourcePageIndex,
    pageSize,
    drawable: clipPolygonToPage(translatePolygon(placement.drawable, dx, dy), pageSize),
    bbox,
    textBoxes: placement.textBoxes
      .map(box => clipBBoxToPage(translateBBox(box, dx, dy), pageSize))
      .filter((box): box is BBox => box !== null),
  }
}

function projectMargin(
  margin: SafeMarginsDebug,
  projection: PreparedPageHandle['projections'][number],
  fallbackBounds: BBox,
): SafeMarginsDebug {
  const dx = projection.sourceRect.x - projection.preparedRect.x
  const dy = projection.sourceRect.y - projection.preparedRect.y
  const pageSize = projectionPageSize(projection)
  const safeBounds = clipBBoxToPage(translateBBox(margin.safeBounds, dx, dy), pageSize) ?? fallbackBounds
  return {
    ...margin,
    margins: marginsWithinBounds(fallbackBounds, safeBounds),
    safeBounds,
    componentBBox: margin.componentBBox ? clipBBoxToPage(translateBBox(margin.componentBBox, dx, dy), pageSize) : null,
    shape: projectShape(margin, dx, dy, pageSize),
  }
}

function projectShape(
  margin: SafeMarginsDebug,
  dx: number,
  dy: number,
  pageSize: readonly [number, number],
): SafeMarginsDebug['shape'] {
  if (!margin.shape) return null
  const bounds = clipBBoxToPage(translateBBox(margin.shape.bounds, dx, dy), pageSize)
  if (!bounds) return null
  const spans = margin.shape.spans
    .map(span => ({ y: span.y + dy, x1: clamp(span.x1 + dx, 0, pageSize[0]), x2: clamp(span.x2 + dx, 0, pageSize[0]) }))
    .filter(span => span.y >= 0 && span.y <= pageSize[1] && span.x2 - span.x1 >= 1)
  return spans.length ? { ...margin.shape, bounds, spans } : null
}

function translateBBox(bbox: BBox, dx: number, dy: number): BBox {
  return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]
}

function translatePolygon(polygon: Polygon, dx: number, dy: number): Polygon {
  return polygon.map(point => [point[0] + dx, point[1] + dy])
}

function clipPolygonToPage(polygon: Polygon, pageSize: readonly [number, number]): Polygon {
  const [width, height] = pageSize
  return polygon.map(point => [clamp(point[0], 0, width), clamp(point[1], 0, height)])
}

function clipBBoxToPage(bbox: BBox, pageSize: readonly [number, number]): BBox | null {
  const clipped = clipBBox(bbox, pageSize)
  return clipped[0] < clipped[2] && clipped[1] < clipped[3] ? clipped : null
}

function marginsWithinBounds(bbox: BBox, bounds: BBox): SafeMarginsDebug['margins'] {
  return {
    top: Math.max(0, bbox[1] - bounds[1]),
    bottom: Math.max(0, bounds[3] - bbox[3]),
    left: Math.max(0, bbox[0] - bounds[0]),
    right: Math.max(0, bounds[2] - bbox[2]),
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value))
}

function pageSizeForSource(
  handle: PreparedPageHandle,
  sourcePageIndex: number,
) {
  const projection = handle.projections.find(
    p => p.sourcePageIndex === sourcePageIndex,
  )
  if (projection) {
    const [width, height] = projectionPageSize(projection)
    return { width, height }
  }
  return handle.size
}

function projectionPageSize(projection: PreparedPageHandle['projections'][number]): readonly [number, number] {
  return projection.sourcePageSize
    ? [projection.sourcePageSize.width, projection.sourcePageSize.height]
    : [projection.sourceRect.width, projection.sourceRect.height]
}
