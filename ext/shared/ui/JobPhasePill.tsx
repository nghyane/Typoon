// Phase chip — compact icon + label badge for queue rows.
//
// Pro pattern: visual language (icon) carries the phase, color carries
// the tone. Queue rows can show 6 phases without text alone forcing
// each row to widen.

import {
  AlertTriangle, Check, Clock, Cog, Download, Loader2, Package, Upload,
} from 'lucide-react'
import type { JobPhase } from '@core/upload/state'
import { cn } from '@shared/lib/cn'

interface PhaseMeta {
  icon:   typeof Clock
  label:  string
  /** background tint */
  bg:     string
  /** foreground (icon + text) */
  fg:     string
  /** spin the icon (for active long-running phases) */
  spin?:  boolean
}

const META: Record<JobPhase, PhaseMeta> = {
  queued:     { icon: Clock,    label: 'Chờ',     bg: 'bg-surface-2',  fg: 'text-text-muted'    },
  fetching:   { icon: Download, label: 'Tải ảnh', bg: 'bg-info-bg',    fg: 'text-info-text'     },
  packing:    { icon: Package,  label: 'Đóng gói', bg: 'bg-info-bg',   fg: 'text-info-text'     },
  uploading:  { icon: Upload,   label: 'Tải lên', bg: 'bg-info-bg',    fg: 'text-info-text'     },
  finalizing: { icon: Cog,      label: 'Xử lý',   bg: 'bg-info-bg',    fg: 'text-info-text', spin: true },
  done:       { icon: Check,    label: 'Xong',    bg: 'bg-success-bg', fg: 'text-success-text'  },
  error:      { icon: AlertTriangle, label: 'Lỗi', bg: 'bg-error-bg',  fg: 'text-error-text'    },
}

const ACTIVE: Set<JobPhase> = new Set(['fetching', 'packing', 'uploading'])


export function JobPhasePill({
  phase, className,
}: {
  phase:     JobPhase
  className?: string
}) {
  const meta = META[phase]
  const Icon = meta.icon
  const showSpinner = meta.spin || (ACTIVE.has(phase) && phase !== 'uploading')
  const SpinnerIcon = Loader2

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 h-5 px-1.5 rounded-full text-[10px] font-medium',
        meta.bg, meta.fg, className,
      )}
    >
      {showSpinner
        ? <SpinnerIcon size={9} className="animate-spin" />
        : <Icon size={9} strokeWidth={2.5} />}
      <span>{meta.label}</span>
    </span>
  )
}
