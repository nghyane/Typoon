// Quota meter — used in Home hero + Settings.
//
// Color shifts from accent → warning → error as usage approaches the
// cap. Reset date is formatted locale-aware.

import { cn } from '@shared/lib/cn'
import type { ApiQuota } from '@shared/api/api'
import { TierBadge } from './TierBadge'

interface Props {
  quota:      ApiQuota
  /** Compact variant for sidebar — single line, no labels. */
  compact?:   boolean
  className?: string
}

const formatResetDate = (iso: string): string => {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      day: 'numeric', month: 'long',
    })
  } catch { return iso }
}

const barTone = (pct: number): string => {
  if (pct >= 0.95) return 'bg-error'
  if (pct >= 0.80) return 'bg-warning'
  return 'bg-accent'
}

export function QuotaMeter({ quota, compact, className }: Props) {
  const { tier, used_chapters, active_jobs, reset_at } = quota
  const pct = Math.min(used_chapters / Math.max(tier.monthly_chapters, 1), 1)
  const pctStr = `${Math.round(pct * 100)}%`

  if (compact) {
    return (
      <div className={cn('flex items-center gap-2 text-xs', className)}>
        <div className="flex-1 h-1 bg-surface-2 rounded-full overflow-hidden">
          <div className={cn('h-full transition-all', barTone(pct))} style={{ width: pctStr }} />
        </div>
        <span className="text-text-muted tabular-nums">
          {used_chapters}/{tier.monthly_chapters}
        </span>
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between text-sm">
        <span className="text-text-muted">Hạn mức tháng này</span>
        <TierBadge tier={tier} />
      </div>

      <div className="space-y-1">
        <div className="h-2 bg-surface-2 rounded-full overflow-hidden">
          <div className={cn('h-full transition-all', barTone(pct))} style={{ width: pctStr }} />
        </div>
        <div className="flex items-center justify-between text-xs text-text-muted tabular-nums">
          <span>
            {used_chapters} / {tier.monthly_chapters} chương · {active_jobs}/{tier.concurrent_jobs} đang chạy
          </span>
          <span>Reset {formatResetDate(reset_at)}</span>
        </div>
      </div>
    </div>
  )
}
