type Variant = 'primary' | 'secondary' | 'ghost'
type Size    = 'sm' | 'md'

const V: Record<Variant, string> = {
  primary:   'bg-(--color-btn-bg) text-(--color-btn-text) hover:opacity-90',
  secondary: 'bg-(--color-bg) text-(--color-text) border border-(--color-border) hover:bg-(--color-surface)',
  ghost:     'text-(--color-text-2) hover:bg-(--color-surface)',
}
const S: Record<Size, string> = {
  sm: 'h-7  px-3   text-xs  rounded-lg  gap-1.5',
  md: 'h-8  px-3.5 text-sm  rounded-lg  gap-2',
}

export function Button({ variant = 'secondary', size = 'md', className = '', children, ...props }:
  { variant?: Variant; size?: Size; className?: string } & React.ButtonHTMLAttributes<HTMLButtonElement>
) {
  return (
    <button
      className={`inline-flex items-center font-medium transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed ${V[variant]} ${S[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}
