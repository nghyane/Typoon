import type { TranslationRequest } from '../domain/run'

type TranslationScope = NonNullable<TranslationRequest['scope']>

export interface SourcePlan {
  readonly statusPages: readonly number[]
  readonly loadOrder: readonly number[]
  readonly prepareOrder: readonly number[]
}

/**
 * Build source orders for a run.
 *
 * loadOrder can prioritize the current page.
 * prepareOrder is allowed to differ because continuous prepare needs
 * source-natural input order even if loading is prioritized.
 */
export function buildSourcePlan(args: {
  readonly pageCount: number
  readonly scope: TranslationScope
  readonly priority?: TranslationRequest['priority']
  readonly preparation: TranslationRequest['preparation']
}): SourcePlan {
  const statusPages = buildScope(args.pageCount, args.scope)
  const prepareOrder = args.preparation.type === 'identity'
    ? args.priority
      ? priorityOrder(statusPages, args.priority.aroundPageIndex)
      : statusPages
    : [...statusPages].sort((a, b) => a - b)

  return {
    statusPages,
    loadOrder: prepareOrder,
    prepareOrder,
  }
}

function priorityOrder(indexes: readonly number[], around: number): number[] {
  const available = new Set(indexes)
  if (!available.has(around)) return [...indexes]

  const maxDistance = Math.max(...indexes.map(index => Math.abs(index - around)))
  const out: number[] = [around]

  for (let distance = 1; distance <= maxDistance; distance++) {
    const next = around + distance
    if (available.has(next)) out.push(next)

    const prev = around - distance
    if (available.has(prev)) out.push(prev)
  }

  return out
}

function buildScope(
  pageCount: number,
  scope: TranslationScope,
): number[] {
  if (scope === 'all') {
    return Array.from({ length: pageCount }, (_, i) => i)
  }
  return [...new Set(scope.filter(i => i >= 0 && i < pageCount))]
}
