import { cn } from '../../lib/cn'

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info' | 'idle'

const STYLES: Record<BadgeVariant, { bg: string; color: string }> = {
  default: { bg: 'var(--color-surface-3)',    color: 'var(--color-text-2)' },
  success: { bg: 'var(--color-success-muted)', color: 'var(--color-success)' },
  warning: { bg: 'var(--color-warning-muted)', color: 'var(--color-warning)' },
  error:   { bg: 'var(--color-error-muted)',   color: 'var(--color-error)' },
  info:    { bg: 'var(--color-info-muted)',     color: 'var(--color-info)' },
  idle:    { bg: 'var(--color-surface-3)',      color: 'var(--color-idle)' },
}

interface BadgeProps {
  variant?: BadgeVariant
  className?: string
  children: React.ReactNode
}

export function Badge({ variant = 'default', className, children }: BadgeProps) {
  const s = STYLES[variant]
  return (
    <span
      className={cn('inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium', className)}
      style={{ background: s.bg, color: s.color }}
    >
      {children}
    </span>
  )
}
