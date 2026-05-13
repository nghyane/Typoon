import { useQuery } from '@tanstack/react-query'
import { useState, useEffect, useRef } from 'react'
import {
  AlertTriangle, Hourglass, Loader2, PauseCircle,
} from 'lucide-react'
import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { cn } from '@shared/lib/cn'

// =============================================================================
// WorkersIndicator — header pill showing current pipeline state.
//
// Visible state priority (highest first), so the user reads ONE
// signal instead of a mixed message:
//
//   1. blocked  — at least one stage is paused waiting for admin.
//                 Amber pill, no spinner.  "Tạm ngưng N"
//   2. failed   — dead-lettered tasks (attempts past MAX_ATTEMPTS).
//                 Rose pill.               "Lỗi N"
//   3. running  — workers actively processing chapters.
//                 Info pill + spinner.     "Đang xử lý N"
//   4. pending  — queued, waiting for an idle worker.
//                 Warning pill.            "N chờ"
//
// Hides entirely when everything is zero.
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
    queryKey: qk.workers(),
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
  const pausedStages = data?.paused_stages ?? []
  const totals = sumTotals(stages)

  // Idle — nothing is happening anywhere. Hide the pill so the
  // header stays uncluttered.
  if (totals.all === 0 && pausedStages.length === 0) return null

  const tone = pickTone(totals, pausedStages.length)
  const { icon, label } = pillFor(tone, totals)

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-2 h-8 px-2.5 rounded-sm cursor-pointer transition-colors',
          toneClasses(tone),
        )}
      >
        {icon}
        <span className="text-xs font-medium tabular">{label}</span>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1.5 w-80 z-50 rounded-md bg-surface shadow-[0_8px_24px_rgb(0,0,0,0.4)] overflow-hidden">
          <header className="px-3.5 py-2.5 border-b border-border-soft">
            <p className="text-sm font-semibold text-text">Hàng đợi dịch</p>
            <p className="text-xs text-text-subtle mt-0.5">
              {totals.running} chạy · {totals.pending} chờ
              {totals.blocked > 0 && ` · ${totals.blocked} tạm ngưng`}
              {totals.failed  > 0 && ` · ${totals.failed} lỗi`}
            </p>
          </header>

          {/* Paused-stage banner takes top billing — this is the
              ONE thing the user needs to know about. */}
          {pausedStages.length > 0 && (
            <div className="px-3.5 py-2.5 border-b border-border-soft bg-amber-500/5">
              <p className="text-xs font-medium text-amber-300 inline-flex items-center gap-1.5">
                <PauseCircle size={12} />
                Hệ thống tạm ngưng
              </p>
              <p className="text-xs text-amber-200/70 mt-1">
                {pausedStages.map((s) => STAGE_LABEL[s] ?? s).join(', ')}
                {' — '}đang chờ quản trị xử lý.
              </p>
            </div>
          )}

          <div className="px-3.5 py-2.5 space-y-2">
            {(['scan', 'translate', 'render'] as const).map((stage) => {
              const s = stages[stage] ?? {
                pending: 0, running: 0, stale: 0, blocked: 0, failed: 0,
              }
              const total = s.pending + s.running + s.stale
                          + s.blocked + s.failed
              if (total === 0) return null
              const isPaused = pausedStages.includes(stage)
              return (
                <div key={stage} className="flex items-center justify-between text-xs">
                  <span
                    className={cn(
                      'inline-flex items-center gap-1.5',
                      isPaused ? 'text-amber-300' : 'text-text-muted',
                    )}
                  >
                    {isPaused && <PauseCircle size={11} />}
                    {STAGE_LABEL[stage]}
                  </span>
                  <div className="flex items-center gap-3">
                    {s.running > 0 && (
                      <Stat icon={<Loader2 size={12} className="animate-spin" />}
                            count={s.running} color="text-info-text" />
                    )}
                    {s.pending > 0 && !isPaused && (
                      <Stat icon={<Hourglass size={12} />}
                            count={s.pending} color="text-warning-text" />
                    )}
                    {s.blocked > 0 && (
                      <Stat icon={<PauseCircle size={12} />}
                            count={s.blocked} color="text-amber-400"
                            title="Tạm ngưng — chờ quản trị" />
                    )}
                    {s.stale > 0 && (
                      <Stat icon={<span className="size-1.5 rounded-full bg-error" />}
                            count={s.stale} color="text-error-text"
                            title="Bị treo, không phản hồi" />
                    )}
                    {s.failed > 0 && (
                      <Stat icon={<AlertTriangle size={12} />}
                            count={s.failed} color="text-rose-400"
                            title="Đã hết lượt thử lại — cần quản trị" />
                    )}
                  </div>
                </div>
              )
            })}

            {totals.stale > 0 && (
              <p className="text-xs text-error-text/80 pt-2 border-t border-border-soft">
                {totals.stale} chương bị treo trên 10 phút — kiểm tra worker.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}


// ── Helpers ───────────────────────────────────────────────────


type StageCounts = {
  pending: number; running: number; stale: number
  blocked: number; failed:  number
}

type Tone = 'blocked' | 'failed' | 'running' | 'pending' | 'stale'


function sumTotals(stages: Record<string, StageCounts>) {
  let pending = 0, running = 0, stale = 0, blocked = 0, failed = 0
  for (const s of Object.values(stages)) {
    pending += s.pending ?? 0
    running += s.running ?? 0
    stale   += s.stale   ?? 0
    blocked += s.blocked ?? 0
    failed  += s.failed  ?? 0
  }
  return {
    pending, running, stale, blocked, failed,
    all: pending + running + stale + blocked + failed,
  }
}


function pickTone(
  t: ReturnType<typeof sumTotals>,
  pausedCount: number,
): Tone {
  if (t.blocked > 0 || pausedCount > 0) return 'blocked'
  if (t.failed  > 0)                    return 'failed'
  if (t.stale   > 0 && t.running === 0) return 'stale'
  if (t.running > 0)                    return 'running'
  return 'pending'
}


function pillFor(tone: Tone, t: ReturnType<typeof sumTotals>) {
  switch (tone) {
    case 'blocked':
      return {
        icon:  <PauseCircle size={14} />,
        label: `Tạm ngưng ${t.blocked || ''}`.trim(),
      }
    case 'failed':
      return {
        icon:  <AlertTriangle size={14} />,
        label: `Lỗi ${t.failed}`,
      }
    case 'stale':
      return {
        icon:  <AlertTriangle size={14} />,
        label: `Treo ${t.stale}`,
      }
    case 'running':
      return {
        icon:  <Loader2 size={14} className="animate-spin" />,
        label: `Đang xử lý ${t.running}`,
      }
    case 'pending':
    default:
      return {
        icon:  <Hourglass size={14} />,
        label: `${t.pending} chờ`,
      }
  }
}


function toneClasses(tone: Tone): string {
  switch (tone) {
    case 'blocked': return 'bg-amber-500/15 text-amber-300 hover:brightness-110'
    case 'failed':  return 'bg-rose-500/15 text-rose-300 hover:brightness-110'
    case 'stale':   return 'bg-error-bg text-error-text hover:brightness-110'
    case 'running': return 'bg-info-bg text-info-text hover:brightness-110'
    case 'pending':
    default:        return 'bg-warning-bg text-warning-text hover:brightness-110'
  }
}


function Stat({
  icon, count, color, title,
}: {
  icon:   React.ReactNode
  count:  number
  color:  string
  title?: string
}) {
  return (
    <span
      className={cn('inline-flex items-center gap-1.5', color)}
      title={title}
    >
      {icon}
      <span className="tabular">{count}</span>
    </span>
  )
}
