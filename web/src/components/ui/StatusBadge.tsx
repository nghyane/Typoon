import { cn } from '../../lib/cn'

type StatusVariant = 'done' | 'running' | 'pending' | 'idle' | 'error'

const CONFIG: Record<StatusVariant, { dot: string; label: string; text: string }> = {
  done:    { dot: 'bg-(--color-done)',    label: 'Hoàn thành',  text: 'text-(--color-done)' },
  running: { dot: 'bg-(--color-running)', label: 'Đang xử lý',  text: 'text-(--color-running)' },
  pending: { dot: 'bg-(--color-pending)', label: 'Chờ xử lý',   text: 'text-(--color-pending)' },
  idle:    { dot: 'bg-(--color-idle)',    label: 'Chưa dịch',   text: 'text-(--color-text-3)' },
  error:   { dot: 'bg-(--color-error)',   label: 'Lỗi',         text: 'text-(--color-error)' },
}

export function StatusBadge({ variant, label, className }: {
  variant: StatusVariant
  label?: string
  className?: string
}) {
  const c = CONFIG[variant]
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-sm', c.text, className)}>
      <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', c.dot)} />
      {label ?? c.label}
    </span>
  )
}
