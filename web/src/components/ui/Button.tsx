import { cn } from '../../lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size    = 'sm' | 'md'

const VARIANT: Record<Variant, string> = {
  primary:   'bg-(--color-accent) text-white hover:bg-(--color-accent-hover)',
  secondary: 'bg-(--color-bg) text-(--color-text-1) border border-(--color-border) hover:bg-(--color-surface)',
  ghost:     'text-(--color-text-2) hover:bg-(--color-surface) hover:text-(--color-text-1)',
  danger:    'bg-(--color-bg) text-(--color-error) border border-(--color-border) hover:bg-(--color-error-bg)',
}

const SIZE: Record<Size, string> = {
  sm: 'px-3 py-1 text-xs rounded-md gap-1.5 h-7',
  md: 'px-3 py-1.5 text-sm rounded-md gap-2 h-8',
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
}

export function Button({ variant = 'secondary', size = 'md', className, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center font-medium transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed',
        VARIANT[variant], SIZE[size], className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}
