import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cn } from '@shared/lib/cn'

// =============================================================================
// Button — single component with explicit `variant` × `size` × `icon` props.
// Replaces the legacy `btn.primary` / `btn.ghost` string-record pattern, which
// conflated "ghost" with "icon-only inside table rows" and used string concat
// for size variants (no type safety, no proper hover/disabled discipline).
// =============================================================================

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type ButtonSize    = 'sm' | 'md' | 'lg'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?:    ButtonSize
  /** square icon-only button — width = size's height */
  icon?:    boolean
  children?: ReactNode
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
  lg: 'h-10 px-4 text-sm',
}

const SIZE_ICON: Record<ButtonSize, string> = {
  sm: 'size-7 p-0',
  md: 'size-8 p-0',
  lg: 'size-10 p-0',
}

const VARIANT: Record<ButtonVariant, string> = {
  primary:   'bg-accent text-accent-fg hover:brightness-110',
  secondary: 'bg-surface-2 text-text hover:bg-interactive-hover',
  ghost:     'bg-transparent text-text-muted hover:text-text hover:bg-hover',
  danger:    'bg-error text-white hover:brightness-110',
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon = false, className, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(BASE, icon ? SIZE_ICON[size] : SIZE[size], VARIANT[variant], className)}
      {...props}
    />
  )
})
