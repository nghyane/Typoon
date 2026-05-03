import { cn } from '../../lib/cn'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
}

const VARIANTS = {
  primary:   { background: 'var(--color-accent)',   color: '#fff',                    hover: 'hover:opacity-90' },
  secondary: { background: 'var(--color-surface-3)', color: 'var(--color-text-1)',    hover: 'hover:opacity-80' },
  ghost:     { background: 'transparent',            color: 'var(--color-text-2)',    hover: 'hover:opacity-80' },
  danger:    { background: 'var(--color-error-muted)', color: 'var(--color-error)',   hover: 'hover:opacity-80' },
}

const SIZES = {
  sm: 'px-2.5 py-1 text-xs rounded-md',
  md: 'px-3.5 py-1.5 text-sm rounded-lg',
}

export function Button({ variant = 'secondary', size = 'md', className, children, ...props }: ButtonProps) {
  const v = VARIANTS[variant]
  return (
    <button
      className={cn(
        'inline-flex items-center gap-1.5 font-medium transition-opacity disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed',
        SIZES[size],
        v.hover,
        className,
      )}
      style={{ background: v.background, color: v.color }}
      {...props}
    >
      {children}
    </button>
  )
}
