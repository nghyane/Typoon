type Variant = 'done' | 'running' | 'pending' | 'idle' | 'error'

const BAR: Record<Variant, string> = {
  done:    'bg-(--color-done)',
  running: 'bg-(--color-running)',
  pending: 'bg-(--color-pending)',
  idle:    'bg-(--color-idle)',
  error:   'bg-(--color-error)',
}

export function ProgressBar({ value, variant = 'idle', className }: {
  value: number
  variant?: Variant
  className?: string
}) {
  return (
    <div className={`h-1.5 rounded-full overflow-hidden bg-(--color-border-subtle) ${className ?? ''}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${value > 0 ? BAR[variant] : ''}`}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}
