import { useQuery } from '@tanstack/react-query'
import { Zap } from 'lucide-react'
import { api } from '@shared/api/api'
import { cn } from '../shared/lib/cn'

const W_COLLAPSED = 60
const NAV_PAD_X   = 8
const NAV_LANE    = W_COLLAPSED - NAV_PAD_X * 2

interface Props {
  collapsed: boolean
}

// Sidebar quota chip — surfaces "X/Y today" so users notice the cap
// before hitting 429. Polls /api/me/quota at 30s; invalidated by
// callers (start/redo/upload mutations) via queryClient.invalidateQueries.
//
// Hidden entirely for admins (their quota is uncapped) and while no
// data is in cache (avoid a flash of "0/0" on first render).
export function SidebarQuota({ collapsed }: Props) {
  const { data } = useQuery({
    queryKey: ['quota'],
    queryFn:  api.getQuota,
    refetchInterval: 30_000,
    refetchOnWindowFocus: true,
    staleTime: 10_000,
  })

  if (!data || data.is_admin) return null

  const { used_day, limit_day, used_hour, limit_hour, in_flight, limit_concurrent } = data

  // Color: green > 50% remaining, amber 10–50%, red < 10% or in_flight at cap.
  const dayPct  = limit_day  > 0 ? used_day  / limit_day  : 0
  const hourPct = limit_hour > 0 ? used_hour / limit_hour : 0
  const peak    = Math.max(dayPct, hourPct)
  const atConcurrentCap = in_flight >= limit_concurrent

  const color =
    atConcurrentCap || peak >= 0.9 ? 'text-rose-400'
    : peak >= 0.5                  ? 'text-amber-400'
    :                                'text-text-muted'

  return (
    <div
      title={
        `Hôm nay: ${used_day}/${limit_day} chương\n` +
        `Trong giờ: ${used_hour}/${limit_hour}\n` +
        `Đang xử lý: ${in_flight}/${limit_concurrent}`
      }
      className={cn(
        'group flex items-center h-8 w-full rounded-sm select-none',
        'text-text-muted',
      )}
    >
      <span
        style={{ width: NAV_LANE }}
        className="h-full flex items-center justify-center shrink-0"
      >
        <Zap size={16} className={color} />
      </span>
      <span
        className="flex-1 min-w-0 truncate pr-2.5 text-[12px] tabular-nums transition-opacity duration-150"
        style={{ opacity: collapsed ? 0 : 1 }}
      >
        <span className={color}>{used_day}</span>
        <span className="text-text-subtle">/{limit_day}</span>
        <span className="text-text-subtle ml-1">hôm nay</span>
      </span>
    </div>
  )
}
