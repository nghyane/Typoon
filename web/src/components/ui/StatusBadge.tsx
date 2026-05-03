type Variant = 'done' | 'running' | 'pending' | 'idle' | 'error'

const DOT: Record<Variant, string> = {
  done:    'bg-(--color-green)',
  running: 'bg-(--color-blue)',
  pending: 'bg-(--color-orange)',
  idle:    'bg-(--color-text-4)',
  error:   'bg-(--color-red)',
}
const COLOR: Record<Variant, string> = {
  done:    'text-(--color-green)',
  running: 'text-(--color-blue)',
  pending: 'text-(--color-orange)',
  idle:    'text-(--color-text-3)',
  error:   'text-(--color-red)',
}
const LABEL: Record<Variant, string> = {
  done:    'Hoàn thành',
  running: 'Đang xử lý',
  pending: 'Chờ xử lý',
  idle:    'Chưa dịch',
  error:   'Lỗi',
}

export function StatusBadge({ variant, label, className = '' }: {
  variant: Variant; label?: string; className?: string
}) {
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm ${COLOR[variant]} ${className}`}>
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT[variant]}`} />
      {label ?? LABEL[variant]}
    </span>
  )
}
