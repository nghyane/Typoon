// State badges — token-driven, accessible, no emoji.

import {
  CircleDot, Clock, Loader2, CheckCircle2, AlertCircle, Archive,
} from 'lucide-react'
import { Badge, type BadgeTone } from '@shared/ui/primitives'
import type { JobState } from '@shared/api/api'

const STATE_LABEL: Record<JobState, string> = {
  init:      'Khởi tạo',
  uploading: 'Đang tải lên',
  pending:   'Đang chờ',
  running:   'Đang xử lý',
  done:      'Đã xong',
  error:     'Lỗi',
  expired:   'Hết hạn',
}

const STATE_TONE: Record<JobState, BadgeTone> = {
  init:      'neutral',
  uploading: 'info',
  pending:   'neutral',
  running:   'info',
  done:      'success',
  error:     'error',
  expired:   'warning',
}

const STATE_ICON: Record<JobState, React.ComponentType<{ size?: number; className?: string }>> = {
  init:      CircleDot,
  uploading: Loader2,
  pending:   Clock,
  running:   Loader2,
  done:      CheckCircle2,
  error:     AlertCircle,
  expired:   Archive,
}

interface Props {
  state:     JobState
  /** Show a spinning icon for transient states (uploading/running). */
  spin?:     boolean
  className?: string
}

export function JobStateBadge({ state, spin = true, className }: Props) {
  const Icon = STATE_ICON[state]
  const isSpinning = spin && (state === 'running' || state === 'uploading')
  return (
    <Badge tone={STATE_TONE[state]} dot={false} className={className}>
      <Icon size={12} className={isSpinning ? 'animate-spin' : ''} />
      {STATE_LABEL[state]}
    </Badge>
  )
}
