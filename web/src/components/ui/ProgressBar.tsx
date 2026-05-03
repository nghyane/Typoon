interface ProgressBarProps {
  value: number
  variant?: 'done' | 'running' | 'pending' | 'idle' | 'error'
  className?: string
}

const TRACK_COLOR = 'bg-(--color-surface-2)'

const BAR: Record<NonNullable<ProgressBarProps['variant']>, string> = {
  done:    'bg-(--color-done)',
  running: 'bg-(--color-running)',
  pending: 'bg-(--color-pending)',
  idle:    'bg-(--color-idle)',
  error:   'bg-(--color-error)',
}

export function ProgressBar({ value, variant = 'idle', className }: ProgressBarProps) {
  return (
    <div className={`h-1.5 rounded-full overflow-hidden ${TRACK_COLOR} ${className ?? ''}`}>
      <div
        className={`h-full rounded-full transition-all duration-300 ${value > 0 ? BAR[variant] : ''}`}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}
