// Thin progress bar with shimmer overlay for indeterminate states.
//
// Pro pattern: 2 px tall (queue density is tight), 250 ms width
// transition, shimmer when bytes total isn't known yet. Used by
// JobRow during fetch / pack / upload / finalize phases.

import { cn } from '@shared/lib/cn'

interface Props {
  /** 0..100. Clamped. */
  pct:           number
  /** Shimmer overlay; fall back when phase has no real byte total. */
  indeterminate?: boolean
  className?:    string
}

export function JobProgressBar({ pct, indeterminate, className }: Props) {
  const clamped = Math.max(0, Math.min(100, pct))
  return (
    <div className={cn('relative h-[2px] rounded-full bg-surface-2 overflow-hidden', className)}>
      <div
        className="absolute inset-y-0 left-0 bg-accent rounded-full transition-[width] duration-250 ease-out"
        style={{ width: `${clamped}%` }}
      />
      {indeterminate && (
        <div
          aria-hidden
          className="absolute inset-0 pointer-events-none bg-gradient-to-r from-transparent via-white/20 to-transparent animate-[ts-shimmer_1.4s_linear_infinite]"
          style={{ backgroundSize: '200% 100%' }}
        />
      )}
    </div>
  )
}
