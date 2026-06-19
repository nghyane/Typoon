import type { TextPlacement, TextRole } from '../domain/planning'
import type { BBox } from '../domain/geometry'
import { type SafeMargins, type SafeMarginsDebug, type SafeShapeProfile } from './backgroundFit'
import { composeLines, type LineComposition, type LineLayoutCandidate } from './lineComposer'
import { textFitRect, drawableRect, type FitRect } from './fitGeometry'
import { bubbleShapeProfile, type BubbleShapeProfile } from './bubbleShape'
import type { FontProfile } from './font'
import type { DomMeasurer } from './textMeasure'
import { pageRenderProfile, textRenderProfile, type RenderLanguageContext, type TextRenderProfile } from './languageProfile'
import { canUseVerticalTypesetting } from './textScript'

export type TypesetDirection = 'horizontal' | 'vertical'

const MIN_FONT_SIZE = 6
const ABS_MAX_FONT_SIZE = 240
const NORMAL_MAX_FONT_SIZE = 96
const MAX_FONT_PAGE_FRACTION = 0.05
const HIERARCHY_MAX_FONT_PAGE_FRACTION = 0.14
const SOURCE_FONT_MIN = 6
const SOURCE_FONT_MAX = 320
const HIERARCHY_MIN_SAMPLE_COUNT = 3
const HIERARCHY_SPREAD_RATIO = 4
const HIERARCHY_HIGH_RATIO = 2.4

type FontIntentReason = 'role-standard' | 'source' | 'fallback-role-median' | 'fallback-geometry'

export interface FitResult {
  readonly text: string
  readonly fontSizePx: number
  readonly lineHeightPx: number
  readonly paddingXPx: number
  readonly paddingYPx: number
  readonly overflow: boolean
  readonly rect: FitRect
  readonly baseRect: FitRect
  readonly maxDomFitPx: number
  readonly capReason: string
  readonly sourceFontPx: number | null
  readonly roleMedianFontPx: number | null
  readonly targetFontPx: number
  readonly fontIntentReason: FontIntentReason
  readonly fitReason: string
  readonly direction: TypesetDirection
  readonly directionReason: string
  readonly layoutCandidate: LineLayoutCandidate
  readonly lineCount: number
  readonly lineScore: number
  readonly maxFill: number
  readonly edgeGuardPx: number
  readonly fontShortSideRatio: number
  readonly expansion: SafeMarginsDebug | null
  readonly safeShapeUsed: boolean
}

interface FontIntent {
  readonly sourceFontPx: number | null
  readonly roleMedianFontPx: number | null
  readonly targetFontPx: number
  readonly reason: FontIntentReason
}

interface PageFontContext {
  readonly roleMedians: ReadonlyMap<TextRole, number>
  readonly allMedianPx: number | null
  readonly pageMaxPx: number
  readonly preserveSourceScale: boolean
}

export function fitLayout(
  placement: TextPlacement,
  text: string,
  sourceText: string | undefined,
  context: PageFontContext,
  font: FontProfile,
  measurer: DomMeasurer,
  preMargin: SafeMarginsDebug | null,
  languageContext?: RenderLanguageContext,
): FitResult {
  const cleanText = normalizeText(text)
  const profile = textRenderProfile(cleanText, languageContext, placement.role, sourceText)
  const drawableBaseRect = drawableRect(placement)
  const baseRect = textFitRect(placement)
  const drawableShapeProfile = shapeProfileForRect(placement, drawableBaseRect, preMargin)
  const fontIntent = fontIntentFor(placement, baseRect, context, profile)
  const direction = pickDirection(placement, cleanText, drawableBaseRect, drawableShapeProfile, fontIntent.targetFontPx, font, measurer)
  const fontWeight = placement.role === 'sfx' ? '800' : '700'

  // Try fit in base rect at target font
  const baseComposition = composeInRect(fontWeight, cleanText, baseRect, fontIntent.targetFontPx, placement, direction, shapeProfileForRect(placement, baseRect, preMargin), font, measurer, profile)

  const expansion = preMargin && !overallBlocked(preMargin)
    ? constrainExpansionToGeometry(preMargin, baseRect, placement, profile)
    : null

  if (!expansion || overallBlocked(expansion)) {
    return toFitResult(
      text, baseRect, baseRect, baseComposition, fontIntent, direction,
      baseComposition.fontSizePx < fontIntent.targetFontPx ? 'shrink' : 'target',
      baseComposition, expansion ?? preMargin, placement, directionReason(placement, baseRect, cleanText), font.lineHeightRatio,
      profile, safeShapeUsedForRect(placement, baseRect, preMargin),
    )
  }

  // Generate expansion candidates & score them.
  // Clamp every candidate to the detected safe bounds so text never
  // renders beyond the actual background component (bubble / text area).
  const candidates = expansionRects(baseRect, expansion.safeBounds, expansion.margins)
  const baseAspect = baseRect.width / Math.max(1, baseRect.height)
  let bestRect = baseRect
  let bestScore = scoreCandidate(baseComposition, fontIntent.targetFontPx, baseRect, baseAspect, baseRect)

  for (const candidateRect of candidates) {
    if (sameRect(candidateRect, baseRect)) continue
    const comp = composeInRect(fontWeight, cleanText, candidateRect, fontIntent.targetFontPx, placement, direction, shapeProfileForRect(placement, candidateRect, expansion), font, measurer, profile)
    const s = scoreCandidate(comp, fontIntent.targetFontPx, candidateRect, baseAspect, baseRect)
    if (s < bestScore || (s === bestScore && candidateRect.width > bestRect.width)) {
      bestScore = s
      bestRect = candidateRect
    }
  }

  const finalComposition = composeInRect(fontWeight, cleanText, bestRect, fontIntent.targetFontPx, placement, direction, shapeProfileForRect(placement, bestRect, expansion), font, measurer, profile)
  const expanded = !sameRect(bestRect, baseRect)
  const overExpanded = expanded && excessiveExpansionFontLift(baseComposition, finalComposition)
  const outputRect = overExpanded ? baseRect : bestRect
  const outputComposition = overExpanded ? baseComposition : finalComposition
  const outputExpanded = expanded && !overExpanded

  return toFitResult(
    text, outputRect, baseRect, outputComposition, fontIntent, direction,
    outputComposition.fontSizePx < fontIntent.targetFontPx || outputComposition.overflow ? 'shrink' : outputExpanded ? 'expanded' : 'target',
    outputComposition, outputExpanded ? expansion : preMargin,
    placement, directionReason(placement, baseRect, cleanText), font.lineHeightRatio,
    profile, safeShapeUsedForRect(placement, outputRect, outputExpanded ? expansion : preMargin),
  )

}

// ── Page-level context ──────────────────────────────────────────────────────

export function pageFontContext(placements: readonly TextPlacement[], pageWidth: number, languageContext?: RenderLanguageContext): PageFontContext {
  const allSamples = placements.map(p => validSourceFontPx(p.fontHint?.sourceFontPx, textFitRect(p), p.role)).filter((px): px is number => px !== null)
  const preserveSourceScale = hasSourceScaleHierarchy(allSamples)
  const roleMedians = new Map<TextRole, number>()
  for (const role of ['dialogue', 'narration', 'sfx'] as const) {
    const samples = placements
      .filter(p => p.role === role)
      .map(p => validSourceFontPx(p.fontHint?.sourceFontPx, textFitRect(p), p.role))
      .filter((px): px is number => px !== null)
    if (samples.length) roleMedians.set(role, Math.round(median(samples)))
  }
  const profile = pageRenderProfile(languageContext)
  return {
    roleMedians,
    allMedianPx: allSamples.length ? Math.round(median(allSamples)) : null,
    pageMaxPx: maxFontForPage(pageWidth, preserveSourceScale, profile),
    preserveSourceScale,
  }
}

// ── Internal ────────────────────────────────────────────────────────────────

interface SizedComposition {
  readonly fontSizePx: number
  readonly maxDomFitPx: number
  readonly capReason: string
  readonly composition: LineComposition
  readonly overflow: boolean
}

function composeInRect(
  fontWeight: string,
  text: string,
  rect: FitRect,
  targetFontPx: number,
  placement: TextPlacement,
  direction: TypesetDirection,
  shapeProfile: BubbleShapeProfile,
  font: FontProfile,
  measurer: DomMeasurer,
  profile: TextRenderProfile,
): SizedComposition {
  if (!text || rect.width < 1 || rect.height < 1) {
    return {
      fontSizePx: MIN_FONT_SIZE,
      maxDomFitPx: MIN_FONT_SIZE,
      capReason: 'empty',
      composition: emptyComp(text),
      overflow: false,
    }
  }

  const hiBound = Math.max(MIN_FONT_SIZE, Math.min(Math.floor(rect.height), ABS_MAX_FONT_SIZE))
  const { fontSizePx, composition, maxDomFitPx } = bestFontFit({
    placement, text, rect, shapeProfile, targetFontPx, direction, hiBound, font, fontWeight, measurer, profile,
  })
  return {
    fontSizePx,
    maxDomFitPx,
    capReason: fontSizePx < targetFontPx ? 'shrink' : 'target',
    composition,
    overflow: !composition.fits || fontSizePx < profile.minReadableFontPx,
  }
}

function bestFontFit(args: {
  readonly placement: TextPlacement
  readonly text: string
  readonly rect: FitRect
  readonly shapeProfile: BubbleShapeProfile
  readonly targetFontPx: number
  readonly direction: TypesetDirection
  readonly hiBound: number
  readonly font: FontProfile
  readonly fontWeight: string
  readonly measurer: DomMeasurer
  readonly profile: TextRenderProfile
}): { readonly fontSizePx: number; readonly composition: LineComposition; readonly maxDomFitPx: number } {
  // Binary search using browser-native measurement (fast, all scripts).
  let lo = MIN_FONT_SIZE
  let hi = Math.min(args.targetFontPx, args.hiBound)
  let maxFitPx = MIN_FONT_SIZE

  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2)
    const pad = visualPadding(mid, args.profile, args.placement.role)
    const innerW = Math.max(1, args.rect.width - pad.x * 2)
    const innerH = Math.max(1, args.rect.height - pad.y * 2)
    const m = args.measurer.measure({ text: args.text, width: innerW, fontSizePx: mid, fontWeight: args.fontWeight })
    if (m.heightPx <= innerH + 0.5 && !m.overflowWidth) {
      maxFitPx = mid
      lo = mid + 1
    } else {
      hi = mid - 1
    }
  }

  const fontSizePx = Math.max(MIN_FONT_SIZE, Math.min(args.targetFontPx, maxFitPx))
  // Compose with DP for shape-aware line breaks. If DP disagrees with measure(),
  // fall back until DP fits (shape constraints may be tighter than rectangular).
  let composition = composeFit(args, fontSizePx)
  let finalPx = fontSizePx
  while (finalPx > MIN_FONT_SIZE && !composition.fits) {
    finalPx -= 1
    composition = composeFit(args, finalPx)
  }
  const comfortFloor = Math.max(args.profile.minReadableFontPx, Math.floor(fontSizePx * comfortShrinkFloorRatio(args.placement.role, args.profile)))
  const maxComfortFill = comfortMaxFill(args.placement.role, args.profile)
  while (finalPx > comfortFloor && composition.fits && composition.maxFill > maxComfortFill) {
    finalPx -= 1
    composition = composeFit(args, finalPx)
  }
  return { fontSizePx: finalPx, composition, maxDomFitPx: maxFitPx }
}

function composeFit(
  args: {
    readonly text: string
    readonly rect: FitRect
    readonly shapeProfile: BubbleShapeProfile
    readonly direction: TypesetDirection
    readonly font: FontProfile
    readonly fontWeight: string
  readonly placement: TextPlacement
  readonly measurer: DomMeasurer
  readonly profile: TextRenderProfile
  },
  fontSizePx: number,
): LineComposition {
  const pad = visualPadding(fontSizePx, args.profile, args.placement.role)
  return composeLines({
    text: args.text,
    width: Math.max(1, args.rect.width - pad.x * 2),
    height: Math.max(1, args.rect.height - pad.y * 2),
    fontSizePx,
    font: args.font,
    fontWeight: args.fontWeight,
    role: args.placement.role,
    direction: args.direction,
    shapeProfile: args.shapeProfile,
    sourceLineCount: sourceLineCountForLayout(args.placement, args.direction),
    measurer: args.measurer,
  })
}

function sourceLineCountForLayout(placement: TextPlacement, direction: TypesetDirection): number | undefined {
  const hint = placement.fontHint
  if (!hint || hint.sourceDirection !== direction) return undefined
  return hint.sourceLineCount
}

function scoreCandidate(
  comp: SizedComposition,
  targetFontPx: number,
  candRect: FitRect,
  baseAspect: number,
  baseRect: FitRect,
): number {
  if (!comp.composition.fits) return 1_000_000
  // Font preservation is primary. Tie-break with aspect ratio stability.
  const shrinkPx = Math.max(0, targetFontPx - comp.fontSizePx)
  const candAspect = candRect.width / Math.max(1, candRect.height)
  const aspectDrift = Math.abs(candAspect - baseAspect)
  const centerPenalty = centerDistanceRatio(candRect, baseRect)
  // Font shrink dominates; aspect and source-center distance are tie-breakers.
  return shrinkPx * 1000 + Math.round(aspectDrift * 100) + Math.round(centerPenalty * 30)
}

function excessiveExpansionFontLift(base: SizedComposition, expanded: SizedComposition): boolean {
  const liftPx = expanded.fontSizePx - base.fontSizePx
  return liftPx > Math.max(8, base.fontSizePx * 0.30)
}

function expansionRects(baseRect: FitRect, safeBounds: BBox, margins: SafeMargins): FitRect[] {
  const results: FitRect[] = [baseRect]
  const vGrow = Math.min(margins.top, margins.bottom)
  const hGrow = Math.min(margins.left, margins.right)

  if (vGrow > 0) {
    results.push(clampRect({
      x: baseRect.x, y: baseRect.y - vGrow,
      width: baseRect.width, height: baseRect.height + vGrow * 2,
      rotationDeg: baseRect.rotationDeg,
    }, safeBounds))
  }
  if (hGrow > 0) {
    results.push(clampRect({
      x: baseRect.x - hGrow, y: baseRect.y,
      width: baseRect.width + hGrow * 2, height: baseRect.height,
      rotationDeg: baseRect.rotationDeg,
    }, safeBounds))
  }
  if (vGrow > 0 && hGrow > 0) {
    results.push(clampRect({
      x: baseRect.x - hGrow, y: baseRect.y - vGrow,
      width: baseRect.width + hGrow * 2, height: baseRect.height + vGrow * 2,
      rotationDeg: baseRect.rotationDeg,
    }, safeBounds))
  }
  return results
}

function constrainExpansionToGeometry(margin: SafeMarginsDebug, baseRect: FitRect, placement: TextPlacement, profile: TextRenderProfile): SafeMarginsDebug {
  const base = fitRectBBox(baseRect)
  const envelope = geometryEnvelope(baseRect, placement, profile)
  const limited = intersectBBoxes(margin.safeBounds, envelope) ?? base
  const safeBounds = unionBBoxes([base, limited])
  const margins = marginsFromBounds(base, safeBounds)
  const hasGrowth = margins.top + margins.right + margins.bottom + margins.left > 0
  return {
    ...margin,
    safeBounds,
    margins,
    reasons: hasGrowth ? margin.reasons : { ...margin.reasons, overall: 'geometry-limited' },
  }
}

function geometryEnvelope(baseRect: FitRect, placement: TextPlacement, profile: TextRenderProfile): BBox {
  const base = fitRectBBox(baseRect)
  const sourceFont = validSourceFontPx(placement.fontHint?.sourceFontPx)
  const fontPx = sourceFont ?? Math.max(MIN_FONT_SIZE, Math.min(baseRect.width, baseRect.height) * 0.45)
  const sourceVertical = placement.fontHint?.sourceDirection === 'vertical'
  const roleScale = placement.role === 'narration' ? 1.4 : placement.role === 'sfx' ? 0.4 : 1
  const verticalBoost = sourceVertical ? 1.18 : 1
  const growX = Math.max(baseRect.width * profile.geometryGrowWidthRatio, fontPx * profile.geometryGrowXEm * verticalBoost * roleScale)
  const growY = Math.max(baseRect.height * profile.geometryGrowHeightRatio, fontPx * profile.geometryGrowYEm * roleScale)
  return clipBBox([base[0] - growX, base[1] - growY, base[2] + growX, base[3] + growY], placement.pageSize)
}

function fitRectBBox(rect: FitRect): BBox {
  return [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height]
}

function intersectBBoxes(a: BBox, b: BBox): BBox | null {
  const x1 = Math.max(a[0], b[0])
  const y1 = Math.max(a[1], b[1])
  const x2 = Math.min(a[2], b[2])
  const y2 = Math.min(a[3], b[3])
  if (x1 >= x2 || y1 >= y2) return null
  return [x1, y1, x2, y2]
}

function unionBBoxes(boxes: readonly BBox[]): BBox {
  return [Math.min(...boxes.map(box => box[0])), Math.min(...boxes.map(box => box[1])), Math.max(...boxes.map(box => box[2])), Math.max(...boxes.map(box => box[3]))]
}

function marginsFromBounds(base: BBox, bounds: BBox): SafeMargins {
  return {
    top: Math.max(0, base[1] - bounds[1]),
    right: Math.max(0, bounds[2] - base[2]),
    bottom: Math.max(0, bounds[3] - base[3]),
    left: Math.max(0, base[0] - bounds[0]),
  }
}

function clipBBox(bbox: BBox, pageSize: readonly [number, number]): BBox {
  return [Math.max(0, bbox[0]), Math.max(0, bbox[1]), Math.min(pageSize[0], bbox[2]), Math.min(pageSize[1], bbox[3])]
}

function centerDistanceRatio(rect: FitRect, baseRect: FitRect): number {
  const rectCx = rect.x + rect.width / 2
  const rectCy = rect.y + rect.height / 2
  const baseCx = baseRect.x + baseRect.width / 2
  const baseCy = baseRect.y + baseRect.height / 2
  const scale = Math.max(1, Math.hypot(baseRect.width, baseRect.height))
  return Math.hypot(rectCx - baseCx, rectCy - baseCy) / scale
}

/** Clamp a FitRect to stay within the given BBox. Assumes rotationDeg ≈ 0. */
function clampRect(rect: FitRect, bounds: BBox): FitRect {
  const right = rect.x + rect.width
  const bottom = rect.y + rect.height

  const clampedX = Math.max(bounds[0], rect.x)
  const clampedY = Math.max(bounds[1], rect.y)
  const clampedRight = Math.min(bounds[2], right)
  const clampedBottom = Math.min(bounds[3], bottom)

  return {
    x: clampedX,
    y: clampedY,
    width: Math.max(1, clampedRight - clampedX),
    height: Math.max(1, clampedBottom - clampedY),
    rotationDeg: rect.rotationDeg,
  }
}

function shapeProfileForRect(
  placement: TextPlacement,
  rect: FitRect,
  margin: SafeMarginsDebug | null,
): BubbleShapeProfile {
  const safeShape = placement.role === 'sfx' ? null : margin?.shape ?? null
  return safeShapeProfileForRect(safeShape, rect) ?? bubbleShapeProfile(placement.drawable, rect)
}

function safeShapeUsedForRect(placement: TextPlacement, rect: FitRect, margin: SafeMarginsDebug | null): boolean {
  if (placement.role === 'sfx') return false
  return safeShapeProfileForRect(margin?.shape ?? null, rect) !== null
}

function safeShapeProfileForRect(shape: SafeShapeProfile | null, rect: FitRect): BubbleShapeProfile | null {
  if (!shape || shape.confidence < 0.6 || shape.spans.length < 2 || Math.abs(rect.rotationDeg) > 0.1) return null
  if (!safeShapeUsableForRect(shape, rect)) return null
  return {
    kind: 'polygon',
    centerX: rect.x + rect.width / 2,
    centerY: rect.y + rect.height / 2,
    rect,
    widthAt: (lineIndex, totalLines) => safeShapeWidthAt(shape, rect, lineIndex, totalLines),
  }
}

function safeShapeWidthAt(shape: SafeShapeProfile, rect: FitRect, lineIndex: number, totalLines: number): number {
  if (totalLines <= 0) return rect.width
  const y = rect.y + ((lineIndex + 0.5) / totalLines) * rect.height
  const band = Math.max(8, rect.height / Math.max(1, totalLines) * 0.55)
  const widths = shape.spans
    .filter(span => Math.abs(span.y - y) <= band && span.x2 > rect.x && span.x1 < rect.x + rect.width)
    .map(span => shapeSpanWidth(span, rect))
    .filter(width => width > 0)
  if (!widths.length) {
    const span = nearestShapeSpan(shape, y)
    return span ? shapeSpanWidth(span, rect) : rect.width
  }
  return Math.max(1, median(widths))
}

function nearestShapeSpan(shape: SafeShapeProfile, y: number): SafeShapeProfile['spans'][number] | null {
  let best: SafeShapeProfile['spans'][number] | null = null
  let bestDistance = Number.POSITIVE_INFINITY
  for (const span of shape.spans) {
    const distance = Math.abs(span.y - y)
    if (distance < bestDistance) {
      best = span
      bestDistance = distance
    }
  }
  return best
}

function safeShapeUsableForRect(shape: SafeShapeProfile, rect: FitRect): boolean {
  const widths = shape.spans
    .filter(span => span.y >= rect.y && span.y <= rect.y + rect.height && span.x2 > rect.x && span.x1 < rect.x + rect.width)
    .map(span => shapeSpanWidth(span, rect) / Math.max(1, rect.width))
    .filter(ratio => ratio > 0)
    .sort((a, b) => a - b)
  if (widths.length < 3) return false
  const lo = quantile(widths, 0.15)
  const med = quantile(widths, 0.50)
  const hi = quantile(widths, 0.85)
  const meaningfulShape = hi - lo >= 0.18 || lo <= 0.72
  const notNoiseNarrow = med >= 0.68 && hi >= 0.82
  return meaningfulShape && notNoiseNarrow
}

function shapeSpanWidth(span: SafeShapeProfile['spans'][number], rect: FitRect): number {
  const x1 = Math.max(rect.x, span.x1)
  const x2 = Math.min(rect.x + rect.width, span.x2)
  return Math.max(0, x2 - x1)
}

function quantile(sortedValues: readonly number[], q: number): number {
  if (!sortedValues.length) return 0
  const index = Math.min(sortedValues.length - 1, Math.max(0, Math.round((sortedValues.length - 1) * q)))
  return sortedValues[index] ?? 0
}

// ── Direction ───────────────────────────────────────────────────────────────
// Ported from typoon/vision/groupers/lens_native.py _infer_text_direction

function pickDirection(
  placement: TextPlacement,
  text: string,
  baseRect: FitRect,
  _shapeProfile: BubbleShapeProfile,
  _targetFontPx: number,
  _font: FontProfile,
  _measurer: DomMeasurer,
): TypesetDirection {
  // Target text drives direction. Latin / Vietnamese / Korean target
  // must be horizontal regardless of source geometry.
  if (!canUseVerticalTypesetting(text)) return 'horizontal'

  // Target is CJK: use source signals to decide.
  if (placement.fontHint?.sourceDirection === 'vertical') return 'vertical'

  const w = Math.max(1, baseRect.width)
  const h = Math.max(1, baseRect.height)

  // 1. Strict aspect: h > 2w — unambiguous tategaki column
  if (h > w * 2.0) return 'vertical'

  // 2. h > w + CJK text
  if (h > w) return 'vertical'

  return 'horizontal'
}

function directionReason(placement: TextPlacement, rect: FitRect, text: string): string {
  if (!canUseVerticalTypesetting(text)) return 'target-non-vertical-script'
  if (placement.fontHint?.sourceDirection === 'vertical') return 'source-vertical'
  if (rect.height > rect.width * 2.0) return 'strict-aspect'
  return 'cjk-geometry'
}

// ── Font intent ─────────────────────────────────────────────────────────────

function fontIntentFor(placement: TextPlacement, baseRect: FitRect, context: PageFontContext, profile: TextRenderProfile): FontIntent {
  const sourceFontPx = validSourceFontPx(placement.fontHint?.sourceFontPx, baseRect, placement.role)
  const roleMedianFontPx = context.roleMedians.get(placement.role) ?? context.allMedianPx
  const placementMaxPx = maxFontForPlacement(placement, baseRect, context.pageMaxPx, profile)

  // Preserve relative size: each bubble keeps its proportion to the page standard.
  if (roleMedianFontPx !== null && sourceFontPx !== null) {
    const target = clampFont(sourceFontPx * profile.fontScale, placementMaxPx)
    return { sourceFontPx, roleMedianFontPx, targetFontPx: target, reason: 'role-standard' }
  }

  if (roleMedianFontPx !== null) {
    return { sourceFontPx: null, roleMedianFontPx, targetFontPx: clampFont(roleMedianFontPx * profile.fontScale, placementMaxPx), reason: 'fallback-role-median' }
  }
  if (sourceFontPx !== null) {
    return { sourceFontPx, roleMedianFontPx: null, targetFontPx: clampFont(sourceFontPx * profile.fontScale, placementMaxPx), reason: 'source' }
  }
  return { sourceFontPx: null, roleMedianFontPx: null, targetFontPx: clampFont(geometryFallback(placement, baseRect) * profile.fontScale, placementMaxPx), reason: 'fallback-geometry' }
}

// ── Geometry fallback ───────────────────────────────────────────────────────

function geometryFallback(placement: TextPlacement, rect: FitRect): number {
  const shortSide = Math.min(rect.width, rect.height)
  const fraction = placement.role === 'sfx' ? 0.55 : placement.role === 'narration' ? 0.24 : 0.28
  return Math.round(shortSide * fraction)
}

function maxFontForPlacement(placement: TextPlacement, rect: FitRect, pageMaxPx: number, profile: TextRenderProfile): number {
  if (placement.role === 'sfx') return pageMaxPx
  const shortSide = Math.max(1, Math.min(rect.width, rect.height))
  const heightFraction = placement.role === 'narration' ? 0.38
    : profile.targetFamily === 'latin' ? 0.42
    : profile.targetFamily === 'hangul' ? 0.44
    : 0.46
  const shortSideFraction = placement.role === 'narration' ? 0.40 : 0.48
  return Math.max(MIN_FONT_SIZE, Math.min(pageMaxPx, rect.height * heightFraction, shortSide * shortSideFraction))
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function overallBlocked(m: SafeMarginsDebug): boolean {
  return m.reasons.overall === 'rotated' || m.reasons.overall === 'no-background' || m.reasons.overall === 'blocked'
}

function sameRect(a: FitRect, b: FitRect): boolean {
  return a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height && a.rotationDeg === b.rotationDeg
}

function validSourceFontPx(fontPx: number | undefined, rect?: FitRect, role?: TextRole): number | null {
  if (!fontPx || fontPx < SOURCE_FONT_MIN || fontPx > SOURCE_FONT_MAX) return null
  if (rect && role !== 'sfx') {
    const shortSide = Math.max(1, Math.min(rect.width, rect.height))
    const maxByGeometry = Math.max(SOURCE_FONT_MIN, Math.min(rect.height * 0.72, shortSide * 0.76))
    if (fontPx > maxByGeometry) return null
  }
  return fontPx
}

function clampFont(fontPx: number, maxPx: number): number {
  return Math.round(clamp(fontPx, MIN_FONT_SIZE, Math.min(ABS_MAX_FONT_SIZE, maxPx)))
}

function maxFontForPage(pageWidth: number, preserveSourceScale: boolean, profile: Pick<TextRenderProfile, 'pageMaxFraction' | 'hierarchyMaxFraction'>): number {
  const fraction = preserveSourceScale ? Math.min(HIERARCHY_MAX_FONT_PAGE_FRACTION, profile.hierarchyMaxFraction) : Math.min(MAX_FONT_PAGE_FRACTION, profile.pageMaxFraction)
  const floorPx = preserveSourceScale ? 64 : 48
  const ceilingPx = preserveSourceScale ? ABS_MAX_FONT_SIZE : NORMAL_MAX_FONT_SIZE
  return clamp(Math.floor(pageWidth * fraction), floorPx, ceilingPx)
}

function hasSourceScaleHierarchy(samples: readonly number[]): boolean {
  if (samples.length < HIERARCHY_MIN_SAMPLE_COUNT) return false
  const sorted = [...samples].sort((a, b) => a - b)
  const minPx = sorted[0] ?? 0
  const maxPx = sorted[sorted.length - 1] ?? 0
  const medianPx = median(sorted)
  if (minPx <= 0 || medianPx <= 0) return false
  return maxPx / minPx >= HIERARCHY_SPREAD_RATIO && maxPx / medianPx >= HIERARCHY_HIGH_RATIO
}

function normalizeText(text: string): string {
  return text
    .split(/\r?\n/u)
    .map(line => line.split(/\s+/u).filter(Boolean).join(' '))
    .filter(Boolean)
    .join(' ')
}

function innerPadding(fontPx: number, profile: TextRenderProfile): { readonly x: number; readonly y: number } {
  return { x: Math.round(fontPx * profile.innerPadXEm), y: Math.round(fontPx * profile.innerPadYEm) }
}

function visualPadding(fontPx: number, profile: TextRenderProfile, role: TextRole): { readonly x: number; readonly y: number } {
  const inner = innerPadding(fontPx, profile)
  const guard = edgeGuardPx(fontPx, profile, role)
  return { x: inner.x + guard, y: inner.y + Math.ceil(guard * 0.75) }
}

function edgeGuardPx(fontPx: number, profile: TextRenderProfile, role: TextRole): number {
  if (role === 'sfx') return Math.round(clamp(fontPx * 0.08, 2, 18))
  const strokeEm = 0.08
  const glyphEm = profile.targetFamily === 'latin' ? 0.08 : 0.06
  const shapeEm = 0.04
  return Math.ceil(clamp(fontPx * (strokeEm + glyphEm + shapeEm), 3, 24))
}

function comfortMaxFill(role: TextRole, profile: TextRenderProfile): number {
  if (role === 'sfx') return 0.98
  if (profile.targetFamily === 'latin') return 0.90
  if (profile.targetFamily === 'hangul') return 0.92
  return 0.94
}

function comfortShrinkFloorRatio(role: TextRole, profile: TextRenderProfile): number {
  if (role === 'sfx') return 0.90
  if (role === 'narration') return 0.82
  if (profile.targetFamily === 'latin') return 0.78
  return 0.82
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n))
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}

function emptyComp(text: string): LineComposition {
  return { text, lines: [], candidate: 'baseline', lineCount: 0, heightPx: 0, overflowHeight: false, overflowWidth: false, fits: true, score: 0, maxFill: 0 }
}

function toFitResult(
  _text: string,
  rect: FitRect,
  baseRect: FitRect,
  _comp: SizedComposition,
  fontIntent: FontIntent,
  direction: TypesetDirection,
  fitReasonValue: string,
  finalComp: SizedComposition,
  expansion: SafeMarginsDebug | null,
  placement: TextPlacement,
  dirReason: string,
  lineHeightRatio: number,
  profile: TextRenderProfile,
  safeShapeUsed: boolean,
): FitResult {
  const padding = visualPadding(finalComp.fontSizePx, profile, placement.role)
  const shortSide = Math.max(1, Math.min(rect.width, rect.height))
  return {
    text: finalComp.composition.text,
    fontSizePx: finalComp.fontSizePx,
    lineHeightPx: finalComp.fontSizePx * lineHeightRatio,
    paddingXPx: padding.x,
    paddingYPx: padding.y,
    overflow: finalComp.overflow,
    rect,
    baseRect,
    maxDomFitPx: finalComp.maxDomFitPx,
    capReason: finalComp.capReason,
    sourceFontPx: fontIntent.sourceFontPx,
    roleMedianFontPx: fontIntent.roleMedianFontPx,
    targetFontPx: fontIntent.targetFontPx,
    fontIntentReason: fontIntent.reason,
    fitReason: fitReasonValue,
    direction,
    directionReason: dirReason,
    layoutCandidate: finalComp.composition.candidate,
    lineCount: finalComp.composition.lineCount,
    lineScore: finalComp.composition.score,
    maxFill: finalComp.composition.maxFill,
    edgeGuardPx: edgeGuardPx(finalComp.fontSizePx, profile, placement.role),
    fontShortSideRatio: finalComp.fontSizePx / shortSide,
    expansion,
    safeShapeUsed,
  }
}
