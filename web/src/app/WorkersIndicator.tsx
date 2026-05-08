import { useQuery } from '@tanstack/react-query'
import { useState, useEffect, useRef } from 'react'
import { Loader2, Hourglass } from 'lucide-react'
import { api } from '@shared/api/api'
import { cn } from '@shared/lib/cn'

// =============================================================================
// QueueIndicator — header pill showing how many chapters are processing.
// User-facing wording (no "worker" or "pipeline" jargon). Hides when idle.
//
// Click → small popover that explains in plain terms which step each item
// is on (Quét bong bóng / Đang dịch / Đang render). Stale items surface
// with a warning so the user knows a chapter is stuck.
// =============================================================================

const STAGE_LABEL: Record<string, string> = {
  scan:      'Quét bong bóng',
  translate: 'Đang dịch',
  render:    'Đang render',
}

export function WorkersIndicator() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const { data } = useQuery({
    queryKey: ['workers'],
    queryFn:  api.workers,
    refetchInterval: 4000,
    staleTime: 1000,
  })

  // Close popover on outside click.
  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  const stages       = data?.stages ?? {}
  const totalRunning = Object.values(stages).reduce((a, s) => a + s.running, 0)
  const totalPending = Object.values(stages).reduce((a, s) => a + s.pending, 0)
  const totalStale   = Object.values(stages).reduce((a, s) => a + s.stale, 0)

  // Idle — hide entirely. No header noise when nothing is happening.
  if (totalRunning + totalPending + totalStale === 0) return null

  // Primary signal — what's running gets priority over pending.
  const showRunning = totalRunning > 0
  const showStale   = totalStale > 0 && !showRunning

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-1.5 h-8 px-2.5 rounded-sm cursor-pointer transition-colors',
          showStale
            ? 'bg-error-bg text-error-text hover:brightness-110'
            : showRunning
            ? 'bg-info-bg text-info-text hover:brightness-110'
            : 'bg-warning-bg text-warning-text hover:brightness-110',
        )}
      >
        {showRunning ? (
          <Loader2 size={13} className="animate-spin" />
        ) : (
          <Hourglass size={13} />
        )}
        <span className="text-xs font-medium tabular">
          {showRunning
            ? `Đang xử lý ${totalRunning}`
            : `${totalPending} chờ`}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-72 z-50 rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] overflow-hidden">
          <header className="px-3.5 py-2.5 border-b border-border-soft">
            <p className="text-sm font-semibold text-text">Đang xử lý</p>
            <p className="text-[11px] text-text-subtle mt-0.5">
              {totalRunning} chương đang chạy · {totalPending} đang chờ
            </p>
          </header>
          <div className="px-3.5 py-2.5 space-y-2">
            {(['scan', 'translate', 'render'] as const).map((stage) => {
              const s = stages[stage] ?? { pending: 0, running: 0, stale: 0 }
              const total = s.pending + s.running + s.stale
              if (total === 0) return null
              return (
                <div key={stage} className="flex items-center justify-between text-xs">
                  <span className="text-text-muted">{STAGE_LABEL[stage]}</span>
                  <div className="flex items-center gap-3">
                    {s.running > 0 && (
                      <span className="inline-flex items-center gap-1.5 text-info-text">
                        <Loader2 size={10} className="animate-spin" />
                        <span className="tabular">{s.running}</span>
                      </span>
                    )}
                    {s.pending > 0 && (
                      <span className="inline-flex items-center gap-1.5 text-warning-text">
                        <Hourglass size={10} />
                        <span className="tabular">{s.pending}</span>
                      </span>
                    )}
                    {s.stale > 0 && (
                      <span className="inline-flex items-center gap-1.5 text-error-text" title="Bị treo, không phản hồi">
                        <span className="size-1.5 rounded-full bg-error" />
                        <span className="tabular">{s.stale}</span>
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
            {totalStale > 0 && (
              <p className="text-[11px] text-error-text/80 pt-2 border-t border-border-soft">
                {totalStale} chương bị treo — thử chạy lại trong project.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
