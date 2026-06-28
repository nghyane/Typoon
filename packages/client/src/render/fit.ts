import type { TextPlacement, TextRole } from '../domain/planning'
import { type SafeMarginsDebug } from './backgroundFit'
import { type FitRect } from './fitGeometry'
import { type LineLayoutCandidate } from './lineComposer'
import { createDomMeasurer, type DomMeasurer } from './textMeasure'
import type { FontProfile } from './font'
import { fitLayout, pageFontContext, type TypesetDirection } from './fitLayout'
import type { RenderLanguageContext } from './languageProfile'

export type { TypesetDirection }

type FontIntentReason = 'role-standard' | 'source' | 'fallback-role-median' | 'fallback-bubble' | 'fallback-geometry'

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
    const results = items.map((item, index) => fitPlacementText(item, index, context, font, measurer))
    return results
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

// ── Cross-placement font size coordination ──────────────────────────
// When multiple placements share the same role but some shrink more
// than others (e.g. Vietnamese translations are longer than CJK source),
// unify to the minimum within each visually-similar cluster.  A tiny
// bubble must not pull down a large one — they are separate visual
// elements and don't need matching font sizes.

function clusterByMaxDomFit(indices: readonly number[], results: readonly CssFitResult[]): number[][] {
  const sorted = [...indices].sort((a, b) => results[a]!.maxDomFitPx - results[b]!.maxDomFitPx)
  const clusters: number[][] = []
  let cluster: number[] = []
  for (const i of sorted) {
    if (!cluster.length) {
      cluster.push(i)
      continue
    }
    const prev = results[cluster[cluster.length - 1]!]!.maxDomFitPx
    const curr = results[i]!.maxDomFitPx
    if (curr / prev <= 1.8) {
      cluster.push(i)
    } else {
      clusters.push(cluster)
      cluster = [i]
    }
  }
  if (cluster.length) clusters.push(cluster)
  return clusters.filter(c => c.length >= 2)
}

export function coordinateRoleFontSizes(
  items: readonly CssFitInput[],
  results: CssFitResult[],
): CssFitResult[] {
  const roles = new Map<TextRole, number[]>()
  for (let i = 0; i < results.length; i++) {
    const role = items[i]!.placement.role
    const list = roles.get(role)
    if (list) list.push(i)
    else roles.set(role, [i])
  }

  for (const [role, indices] of roles) {
    if (role === 'sfx' || indices.length < 2) continue
    // Only coordinate placements that SHRUNK below their target.
    // Exclude placements where the bubble itself is too small (maxDomFitPx ≈ fontSizePx)
    // — those are geometry-constrained, not text-length-constrained.
    const shrunk = indices.filter(i => {
      const r = results[i]!
      return r.fontSizePx < r.targetFontPx && r.fontSizePx >= r.maxDomFitPx * 0.85
    })
    if (shrunk.length < 2) continue

    // Cluster by similar available space; coordinate only within each cluster.
    for (const cluster of clusterByMaxDomFit(shrunk, results)) {
      const fonts = cluster.map(i => results[i]!.fontSizePx)
      // Unify toward the smallest, but don't let one cramped bubble drag the
      // rest far below their own fit: keep the unified size within 15% of the
      // cluster's largest.  Consistency must not mean "everyone shrinks to the
      // smallest" — that fights the source-proportional size the user expects.
      const unifiedFont = Math.max(Math.min(...fonts), Math.round(Math.max(...fonts) * 0.85))
      for (const i of cluster) {
        const r = results[i]!
        if (r.fontSizePx <= unifiedFont) continue
        const minFont = unifiedFont
        const scale = minFont / r.fontSizePx
        results[i] = {
          ...r,
          fontSizePx: minFont,
          lineHeightPx: Math.round(r.lineHeightPx * scale),
          paddingXPx: Math.round(r.paddingXPx * scale),
          paddingYPx: Math.round(r.paddingYPx * scale),
          fitReason: `${r.fitReason}/uniform`,
        }
      }
    }
  }
  return results
}
