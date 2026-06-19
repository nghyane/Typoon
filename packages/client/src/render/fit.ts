import type { TextPlacement } from '../domain/planning'
import { type SafeMarginsDebug } from './backgroundFit'
import { type FitRect } from './fitGeometry'
import { type LineLayoutCandidate } from './lineComposer'
import { createDomMeasurer, type DomMeasurer } from './textMeasure'
import type { FontProfile } from './font'
import { fitLayout, pageFontContext, type TypesetDirection } from './fitLayout'
import type { RenderLanguageContext } from './languageProfile'

export type { TypesetDirection }

type FontIntentReason = 'role-standard' | 'source' | 'fallback-role-median' | 'fallback-geometry'

export interface CssFitInput {
  readonly placement: TextPlacement
  readonly text: string
  readonly sourceText?: string
}

export interface CssFitResult {
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
  readonly desiredFontSizePx: number
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

interface PageFitContext {
  readonly pageSize: readonly [number, number]
  readonly fontCtx: ReturnType<typeof pageFontContext>
  readonly placementMargins?: readonly SafeMarginsDebug[]
  readonly languageContext?: RenderLanguageContext
}

export function fitPageText(
  items: readonly CssFitInput[],
  pageSize: readonly [number, number],
  font: FontProfile,
  placementMargins?: readonly SafeMarginsDebug[],
  fontContextPlacements?: readonly TextPlacement[],
  languageContext?: RenderLanguageContext,
): CssFitResult[] {
  const measurer = createDomMeasurer(font)
  try {
    const placements = items.map(item => item.placement)
    const fontCtx = pageFontContext(fontContextPlacements?.length ? fontContextPlacements : placements, pageSize[0], languageContext)
    const context: PageFitContext = { pageSize, fontCtx, placementMargins, languageContext }
    return items.map((item, index) => fitPlacementText(item, index, context, font, measurer))
  } finally {
    measurer.destroy()
  }
}

function fitPlacementText(
  input: CssFitInput,
  index: number,
  context: PageFitContext,
  font: FontProfile,
  measurer: DomMeasurer,
): CssFitResult {
  const preMargin = context.placementMargins?.[index] ?? null
  const result = fitLayout(input.placement, input.text, input.sourceText, context.fontCtx, font, measurer, preMargin, context.languageContext)
  return {
    text: result.text,
    fontSizePx: result.fontSizePx,
    lineHeightPx: result.lineHeightPx,
    paddingXPx: result.paddingXPx,
    paddingYPx: result.paddingYPx,
    overflow: result.overflow,
    rect: result.rect,
    baseRect: result.baseRect,
    maxDomFitPx: result.maxDomFitPx,
    capReason: result.capReason,
    desiredFontSizePx: result.targetFontPx,
    sourceFontPx: result.sourceFontPx,
    roleMedianFontPx: result.roleMedianFontPx,
    targetFontPx: result.targetFontPx,
    fontIntentReason: result.fontIntentReason,
    fitReason: result.fitReason,
    direction: result.direction,
    directionReason: result.directionReason,
    layoutCandidate: result.layoutCandidate,
    lineCount: result.lineCount,
    lineScore: result.lineScore,
    maxFill: result.maxFill,
    edgeGuardPx: result.edgeGuardPx,
    fontShortSideRatio: result.fontShortSideRatio,
    expansion: result.expansion,
    safeShapeUsed: result.safeShapeUsed,
  }
}
