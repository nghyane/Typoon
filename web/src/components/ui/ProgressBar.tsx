import { cn } from '../../lib/cn'

interface ProgressBarProps {
  value: number   // 0–100
  variant?: 'default' | 'success' | 'warning' | 'error'
  className?: string
}

const COLORS = {
  default: 'var(--color-accent)',
  success: 'var(--color-success)',
  warning: 'var(--color-warning)',
  error:   'var(--color-error)',
}

export function ProgressBar({ value, variant = 'default', className }: ProgressBarProps) {
  return (
    <div
      className={cn('h-1.5 rounded-full overflow-hidden', className)}
      style={{ background: 'var(--color-surface-3)' }}
    >
      <div
        className="h-full rounded-full transition-all duration-300"
        style={{ width: `${Math.min(100, Math.max(0, value))}%`, background: COLORS[variant] }}
      />
    </div>
  )
}
