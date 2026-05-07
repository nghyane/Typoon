import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { Activity, Cpu } from 'lucide-react'
import { api } from '../lib/api'
import { cn } from '../lib/cn'

const STAGE_LABEL: Record<string, string> = {
  scan:      'Quét',
  translate: 'Dịch',
  render:    'Render',
}

export function WorkersIndicator() {
  const [open, setOpen] = useState(false)

  const { data } = useQuery({
    queryKey: ['workers'],
    queryFn:  api.workers,
    refetchInterval: 4000,
    staleTime: 1000,
  })

  const stages = data?.stages ?? {}
  const totalRunning = Object.values(stages).reduce((a, s) => a + s.running, 0)
  const totalPending = Object.values(stages).reduce((a, s) => a + s.pending + s.stale, 0)

  const tone =
    totalRunning > 0 ? 'running' : totalPending > 0 ? 'pending' : 'idle'

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        title="Trạng thái workers"
        className={cn(
          'flex items-center gap-1.5 h-8 px-2.5 rounded-lg cursor-pointer transition-colors border',
          tone === 'running' && 'bg-blue-50 border-blue-100 text-blue-700 hover:bg-blue-100',
          tone === 'pending' && 'bg-amber-50 border-amber-100 text-amber-700 hover:bg-amber-100',
          tone === 'idle'    && 'bg-white   border-zinc-200 text-zinc-500 hover:bg-zinc-50',
        )}
      >
        <Activity size={13} className={cn(tone === 'running' && 'animate-pulse')} />
        <span className="text-xs font-medium tabular-nums">
          {totalRunning > 0 ? `${totalRunning} chạy` : totalPending > 0 ? `${totalPending} chờ` : 'Rảnh'}
        </span>
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setOpen(false)}
          />
          <div className="absolute right-0 top-full mt-1.5 w-72 z-50 rounded-xl border border-zinc-200 bg-white shadow-[0_8px_24px_rgb(0,0,0,0.08)] overflow-hidden">
            <header className="px-3.5 py-2.5 border-b border-zinc-100 flex items-center gap-2">
              <Cpu size={13} className="text-zinc-400" />
              <span className="text-xs font-semibold text-zinc-700 tracking-wide">PIPELINE</span>
              <span className="ml-auto text-[10px] text-zinc-400 tabular-nums">
                {data?.active_workers.length ?? 0} worker
              </span>
            </header>
            <div className="px-3.5 py-2 space-y-2">
              {(['scan', 'translate', 'render'] as const).map((stage) => {
                const s = stages[stage] ?? { pending: 0, running: 0, stale: 0 }
                const total = s.pending + s.running + s.stale
                return (
                  <div key={stage} className="flex items-center justify-between text-xs">
                    <span className="text-zinc-600 font-medium w-20">{STAGE_LABEL[stage]}</span>
                    <div className="flex-1 flex items-center gap-2 ml-2">
                      {s.running > 0 && (
                        <span className="inline-flex items-center gap-1 text-blue-600">
                          <span className="size-1.5 rounded-full bg-blue-500 animate-pulse" />
                          <span className="tabular-nums">{s.running}</span>
                        </span>
                      )}
                      {s.pending > 0 && (
                        <span className="inline-flex items-center gap-1 text-amber-600">
                          <span className="size-1.5 rounded-full bg-amber-400" />
                          <span className="tabular-nums">{s.pending}</span>
                        </span>
                      )}
                      {s.stale > 0 && (
                        <span className="inline-flex items-center gap-1 text-red-600" title="Worker không phản hồi">
                          <span className="size-1.5 rounded-full bg-red-500" />
                          <span className="tabular-nums">{s.stale}</span>
                        </span>
                      )}
                      {total === 0 && (
                        <span className="text-zinc-300">—</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
