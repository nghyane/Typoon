// Pipeline stage checklist — visualizes job progress.

import { Check, Loader2, Circle, AlertCircle } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import type { ApiJob } from '@shared/api/api'

const STAGES_TRANSLATE = ['prepare', 'scan', 'brief', 'translate', 'typeset', 'finalize'] as const
const STAGES_ANALYZE   = ['prepare', 'scan', 'brief', 'finalize'] as const

type Stage = typeof STAGES_TRANSLATE[number] | typeof STAGES_ANALYZE[number]

const STAGE_LABEL: Record<Stage, string> = {
  prepare:  'Chuẩn bị',
  scan:     'Quét bong bóng',
  brief:    'Phân tích bối cảnh',
  translate:'Dịch',
  typeset:  'Sắp chữ',
  finalize: 'Hoàn tất',
}

interface Props {
  job: ApiJob
}

export function StageChecklist({ job }: Props) {
  const stages: readonly Stage[] = job.kind === 'analyze' ? STAGES_ANALYZE : STAGES_TRANSLATE
  const cur    = job.progress_stage as Stage | null
  const curIdx = cur ? stages.indexOf(cur) : -1

  return (
    <ol className="space-y-1.5 text-sm">
      {stages.map((s, i) => {
        const done    = job.state === 'done' || (curIdx > i)
        const active  = job.state === 'running' && curIdx === i
        const failed  = job.state === 'error' && curIdx === i

        let Icon: React.ComponentType<{ size?: number; className?: string }> = Circle
        let iconClass = 'text-text-subtle'
        if (failed)       { Icon = AlertCircle; iconClass = 'text-error-text' }
        else if (active)  { Icon = Loader2;     iconClass = 'text-accent animate-spin' }
        else if (done)    { Icon = Check;       iconClass = 'text-success' }

        return (
          <li
            key={s}
            className={cn(
              'flex items-center gap-2.5',
              active ? 'text-text' : done ? 'text-text-muted' : 'text-text-subtle',
            )}
          >
            <Icon size={14} className={cn('flex-none', iconClass)} />
            <span>{STAGE_LABEL[s]}</span>

            {active && job.progress_index !== null && job.progress_total !== null && (
              <span className="text-xs text-text-subtle ml-auto tabular-nums">
                {job.progress_index}/{job.progress_total}
              </span>
            )}
          </li>
        )
      })}
    </ol>
  )
}
