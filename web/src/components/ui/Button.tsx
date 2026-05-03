import { cn } from '../../lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost'
type Size = 'sm' | 'md'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
}

const VARIANTS: Record<Variant, string> = {
  primary:   'bg-(--color-accent) text-(--color-accent-text) hover:bg-(--color-accent-hover)',
  secondary: 'bg-(--color-bg) text-(--color-text-1) border border-(--color-border) hover:bg-(--color-surface-1)',
  ghost:     'bg-transparent text-(--color-text-2) hover:bg-(--color-surface-1)',
}

const SIZES: Record<Size, string> = {
  sm: 'px-3 py-1.5 text-xs rounded-lg gap-1.5',
  md: 'px-4 py-2 text-sm rounded-xl gap-2',
}

export function Button({ variant = 'secondary', size = 'md', className, children, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        'inline-flex items-center font-medium transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed',
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}
