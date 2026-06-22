// reader/pageScheduler.ts — page-anchored, viewport-aware, resilient scheduling.
//
// Identity is the page index (reflow-invariant). On top of that identity we add
// per-page retry/skip resilience. A page is "pickable" when it is pending or a
// failed attempt below the retry budget; the nearest page to the viewport
// center is processed first.

import type { PageScanUnit } from '../domain/pageScan'
import type { MeasuredPage } from './pageScan'

export type PageStatus =
  | { readonly kind: 'pending' }
  | { readonly kind: 'processing' }
  | { readonly kind: 'done' }
  | { readonly kind: 'failed'; readonly attempts: number; readonly error: string }

export interface VisibleRange {
  readonly top: number
  readonly bottom: number
  readonly center: number
}

export class PageScheduler {
  private status = new Map<number, PageStatus>()

  /** Reset to the latest unit set, preserving status by page index. */
  reset(units: readonly PageScanUnit[]): void {
    const next = new Map<number, PageStatus>()
    for (const unit of units) next.set(unit.pageIndex, this.status.get(unit.pageIndex) ?? { kind: 'pending' })
    this.status = next
  }

  markProcessing(pageIndex: number): void { this.status.set(pageIndex, { kind: 'processing' }) }
  markDone(pageIndex: number): void { this.status.set(pageIndex, { kind: 'done' }) }

  markFailed(pageIndex: number, error: string): number {
    const prev = this.status.get(pageIndex)
    const attempts = (prev?.kind === 'failed' ? prev.attempts : 0) + 1
    this.status.set(pageIndex, { kind: 'failed', attempts, error })
    return attempts
  }

  /** Pick the nearest-to-viewport page that is pending or retryable. */
  next(
    units: readonly PageScanUnit[],
    measured: ReadonlyMap<number, MeasuredPage>,
    visible: VisibleRange,
    maxAttempts: number,
  ): PageScanUnit | null {
    const pickable = units.filter(unit => {
      const s = this.status.get(unit.pageIndex) ?? { kind: 'pending' }
      return s.kind === 'pending' || (s.kind === 'failed' && s.attempts < maxAttempts)
    })
    if (!pickable.length) return null
    const inView = pickable.filter(unit => {
      const m = measured.get(unit.pageIndex)
      return m && m.domTop < visible.bottom && visible.top < m.domTop + m.domHeight
    })
    const pool = inView.length ? inView : pickable
    return [...pool].sort((x, y) => distance(x, measured, visible.center) - distance(y, measured, visible.center))[0] ?? null
  }

  progress(): { done: number; total: number; failed: number } {
    let done = 0
    let failed = 0
    for (const s of this.status.values()) {
      if (s.kind === 'done') done += 1
      else if (s.kind === 'failed') failed += 1
    }
    return { done, total: this.status.size, failed }
  }

  isComplete(maxAttempts: number): boolean {
    if (this.status.size === 0) return false
    for (const s of this.status.values()) {
      if (s.kind === 'done') continue
      if (s.kind === 'failed' && s.attempts >= maxAttempts) continue
      return false
    }
    return true
  }

  clear(): void { this.status.clear() }
}

function distance(unit: PageScanUnit, measured: ReadonlyMap<number, MeasuredPage>, center: number): number {
  const m = measured.get(unit.pageIndex)
  if (!m) return Number.POSITIVE_INFINITY
  return Math.abs(m.domTop + m.domHeight / 2 - center)
}
