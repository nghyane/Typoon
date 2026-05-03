type Variant = 'done' | 'running' | 'pending' | 'idle' | 'error' | 'purple'

const BAR: Record<Variant, string> = {
  done:    'bg-(--color-green)',
  running: 'bg-(--color-blue)',
  pending: 'bg-(--color-orange)',
  idle:    'bg-(--color-text-4)',
  error:   'bg-(--color-red)',
  purple:  'bg-(--color-purple)',
}

export function ProgressBar({ value, variant = 'idle', className = '' }: {
  value: number; variant?: Variant; className?: string
}) {
  return (
    <div className={`h-1.5 rounded-full bg-(--color-surface-2) overflow-hidden ${className}`}>
      <div
        className={`h-full rounded-full transition-all duration-500 ${BAR[variant]}`}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}
