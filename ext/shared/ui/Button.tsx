import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cn } from '@shared/lib/cn'

// Slim Button — same variants as web/src/shared/ui/Button.tsx but only
// the sizes/variants the popup actually uses (sm, md; primary, ghost,
// danger). When new needs appear, lift more from the web copy.

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type ButtonSize    = 'sm' | 'md'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?:    ButtonSize
}

const BASE =
  'inline-flex items-center justify-center gap-1.5 rounded-sm font-medium ' +
  'transition-[background-color,color,filter] duration-150 cursor-pointer ' +
  'disabled:opacity-50 disabled:cursor-not-allowed ' +
  'whitespace-nowrap select-none ' +
  'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'

const SIZE: Record<ButtonSize, string> = {
  sm: 'h-7 px-2.5 text-xs',
  md: 'h-8 px-3 text-sm',
}

const VARIANT: Record<ButtonVariant, string> = {
  primary:   'bg-accent text-accent-fg hover:brightness-110',
  secondary: 'bg-surface-2 text-text hover:bg-interactive-hover',
  ghost:     'bg-transparent text-text-muted hover:text-text hover:bg-hover',
  danger:    'bg-error text-white hover:brightness-110',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', className, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(BASE, SIZE[size], VARIANT[variant], className)}
      {...props}
    />
  )
})
