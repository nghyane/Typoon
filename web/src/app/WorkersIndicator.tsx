import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import {
  AlertTriangle, Hourglass, Loader2, PauseCircle,
} from 'lucide-react'
import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { useSession } from '@features/auth/session'
import { cn } from '@shared/lib/cn'

// =============================================================================
// WorkersIndicator — header pill, ADMIN ONLY.
//
// Regular users get per-chapter activity via their library cards
// (`translating_summary` on each entry), which answers "is my
// chapter done yet" with the right scope — only their manga, with
// titles and a click-to-resume affordance. The system-wide pill
// here is operator context, not consumer context, so we render
// nothing for non-admins.
//
// For admins, the pill is a status-only widget: visible iff there
// is something to surface. When the pipeline is fully idle the
// pill hides — admins reach /admin/ops via the sidebar entry
// (Sidebar.tsx, NAV_FOOT_ADMIN). The pill is just a fast-path
// "click here to handle the live incident".
//
// Visible-state priority (highest first):
//
//   1. blocked  — stage paused waiting for operator.
//                 Amber pill.              "Tạm ngưng N"
//   2. failed   — dead-letter not yet acknowledged via draft.error.
//                 Rose pill.               "Lỗi N"
//   3. running  — workers active.
//                 Info pill + spinner.     "Đang xử lý N"
//   4. pending  — queued, no idle worker.
//                 Warning pill.            "N chờ"
//   5. stale    — claim older than STALE_CLAIM_INTERVAL.
//                 Error pill.              "Treo N"
// =============================================================================

export function WorkersIndicator() {
  const { user } = useSession()
  const navigate = useNavigate()
  const isAdmin = !!user?.is_admin

  // Skip the query entirely for non-admins: no point polling /workers
  // every 4s for data we'll never render.
  const { data } = useQuery({
    queryKey:        qk.workers(),
    queryFn:         api.workers,
    refetchInterval: 4000,
    staleTime:       1000,
    enabled:         isAdmin,
  })

  if (!isAdmin) return null

  const stages       = data?.stages ?? {}
  const pausedStages = data?.paused_stages ?? []
  const totals = sumTotals(stages)

  // Nothing to surface → hide. Admin still has the sidebar entry.
  if (totals.all === 0 && pausedStages.length === 0) return null

  const tone = pickTone(totals, pausedStages.length)
  const { icon, label } = pillFor(tone, totals)

  return (
    <button
      onClick={() => void navigate({ to: '/admin/ops' })}
      title="Mở quản trị pipeline"
      className={cn(
        'flex items-center gap-2 h-8 px-2.5 rounded-sm cursor-pointer transition-colors',
        toneClasses(tone),
      )}
    >
      {icon}
      <span className="text-xs font-medium tabular">{label}</span>
    </button>
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
