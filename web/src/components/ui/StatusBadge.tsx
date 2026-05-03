import { cn } from '../../lib/cn'

type StatusVariant = 'done' | 'running' | 'pending' | 'idle' | 'error'

const DOT: Record<StatusVariant, string> = {
  done:    'bg-(--color-done)',
  running: 'bg-(--color-running)',
  pending: 'bg-(--color-pending)',
  idle:    'bg-(--color-idle)',
  error:   'bg-(--color-error)',
}

const LABEL: Record<StatusVariant, string> = {
  done:    'Hoàn thành',
  running: 'Đang xử lý',
  pending: 'Chờ xử lý',
  idle:    'Chưa dịch',
  error:   'Lỗi',
}

interface StatusBadgeProps {
  variant: StatusVariant
  label?: string
  className?: string
}

export function StatusBadge({ variant, label, className }: StatusBadgeProps) {
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-sm text-(--color-text-2)', className)}>
      <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', DOT[variant])} />
      {label ?? LABEL[variant]}
    </span>
  )
}
