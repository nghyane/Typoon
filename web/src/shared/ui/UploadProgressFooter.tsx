import { Check, Loader2, Package, Upload as UploadIcon } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { UploadProgress } from '@typoon/upload-sdk'

// =============================================================================
// UploadProgressFooter — full-bleed footer that replaces action buttons
// while a chapter upload is in flight.
//
// Pro pattern (Stripe Checkout / Vercel deploy / Linear async):
//   - 3 phase pills with progressive checkmarks (no text-only label).
//   - Single dominant progress bar (3 px) with smooth width transition
//     and a shimmer overlay during indeterminate phases.
//   - Right-aligned numeric stat: pct + speed + ETA, tabular-nums so
//     digits don't jitter.
//   - Persistent close affordance handled by the modal X — this footer
//     never traps the user.
// =============================================================================

const PHASES = ['packing', 'uploading', 'finalizing'] as const
type Phase = typeof PHASES[number]

const PHASE_META: Record<Phase, { label: string; icon: typeof Package }> = {
  packing:    { label: 'Đóng gói',  icon: Package    },
  uploading:  { label: 'Tải lên',   icon: UploadIcon },
  finalizing: { label: 'Xử lý',     icon: Loader2    },
}

interface Props {
  progress: UploadProgress
}

export function UploadProgressFooter({ progress }: Props) {
  const { phase, bytesSent, bytesTotal, partsSent, partsTotal, speedBps, etaSeconds } = progress

  const phaseIdx = PHASES.indexOf(phase)
  const indeterminate = phase !== 'uploading'

  // Bar width:
  //   uploading → real byte fraction
  //   packing   → 8 % to show motion (full shimmer overlay does the work)
  //   finalizing → 100 % so the bar reads as "done, just polishing"
  const pct = phase === 'uploading' && bytesTotal > 0
    ? Math.min(100, (bytesSent / bytesTotal) * 100)
    : phase === 'finalizing'
      ? 100
      : 8

  return (
    <div className="px-5 py-3">
      {/* Phase indicator + numeric stats — one line, tabular */}
      <div className="flex items-center justify-between gap-4 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          {PHASES.map((p, i) => {
            const Icon = PHASE_META[p].icon
            const done   = i < phaseIdx
            const active = i === phaseIdx
            const spin   = active && p === 'finalizing'
            return (
              <div key={p} className="flex items-center gap-1.5">
                <span
                  className={cn(
                    'size-5 rounded-full grid place-items-center transition-colors',
                    done   && 'bg-success text-white',
                    active && 'bg-accent text-accent-fg',
                    !done && !active && 'bg-surface-2 text-text-subtle',
                  )}
                >
                  {done
                    ? <Check size={11} strokeWidth={3} />
                    : <Icon size={11} className={cn(spin && 'animate-spin')} />}
                </span>
                <span
                  className={cn(
                    'text-xs font-medium tabular truncate',
                    active ? 'text-text' : done ? 'text-text-muted' : 'text-text-subtle',
                  )}
                >
                  {PHASE_META[p].label}
                </span>
                {i < PHASES.length - 1 && (
                  <span
                    aria-hidden
                    className={cn(
                      'h-px w-4 transition-colors',
                      done ? 'bg-success' : 'bg-border-soft',
                    )}
                  />
                )}
              </div>
            )
          })}
        </div>

        <span className="text-xs tabular text-text-muted shrink-0">
          {phase === 'uploading' && bytesTotal > 0
            ? <UploadingStats
                pct={pct}
                bytesSent={bytesSent}
                bytesTotal={bytesTotal}
                partsSent={partsSent}
                partsTotal={partsTotal}
                speedBps={speedBps}
                etaSeconds={etaSeconds}
              />
            : phase === 'packing'
              ? <span className="text-text-subtle">Đang nén…</span>
              : <span className="text-text-subtle">Engine đang chuẩn bị…</span>}
        </span>
      </div>

      {/* Progress bar — 3 px tall, shimmer overlay when indeterminate */}
      <div className="relative h-[3px] rounded-full bg-surface-2 overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 bg-accent rounded-full transition-[width] duration-300 ease-out"
          style={{ width: `${pct}%` }}
        />
        {indeterminate && (
          <div
            aria-hidden
            className={cn(
              'absolute inset-0 pointer-events-none',
              'bg-gradient-to-r from-transparent via-white/15 to-transparent',
              'animate-[ts-shimmer_1.4s_linear_infinite]',
            )}
            style={{ backgroundSize: '200% 100%' }}
          />
        )}
      </div>
    </div>
  )
}


function UploadingStats({
  pct, bytesSent, bytesTotal, partsSent, partsTotal, speedBps, etaSeconds,
}: {
  pct: number; bytesSent: number; bytesTotal: number
  partsSent: number; partsTotal: number
  speedBps?: number; etaSeconds?: number
}) {
  return (
    <span className="flex items-center gap-2">
      <span className="text-text font-medium">{Math.round(pct)}%</span>
      <span className="text-text-subtle/60">·</span>
      <span title={`${bytesSent.toLocaleString()} / ${bytesTotal.toLocaleString()} bytes`}>
        {fmtSize(bytesSent)} / {fmtSize(bytesTotal)}
      </span>
      {partsTotal > 0 && (
        <>
          <span className="text-text-subtle/60">·</span>
          <span>{partsSent}/{partsTotal} phần</span>
        </>
      )}
      {speedBps !== undefined && speedBps > 0 && (
        <>
          <span className="text-text-subtle/60">·</span>
          <span>{fmtSpeed(speedBps)}</span>
        </>
      )}
      {etaSeconds !== undefined && etaSeconds > 0 && (
        <>
          <span className="text-text-subtle/60">·</span>
          <span>còn {fmtEta(etaSeconds)}</span>
        </>
      )}
    </span>
  )
}


function fmtSize(b: number): string {
  if (b < 1024)               return `${b} B`
  if (b < 1024 * 1024)        return `${(b / 1024).toFixed(0)} KB`
  if (b < 1024 * 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`
  return `${(b / 1024 / 1024 / 1024).toFixed(2)} GB`
}

function fmtSpeed(bps: number): string {
  if (bps < 1024)        return `${bps.toFixed(0)} B/s`
  if (bps < 1024 * 1024) return `${(bps / 1024).toFixed(0)} KB/s`
  return `${(bps / 1024 / 1024).toFixed(1)} MB/s`
}

function fmtEta(s: number): string {
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  return `${m}m${r.toString().padStart(2, '0')}s`
}
