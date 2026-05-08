import { type ReactNode } from 'react'
import { cn } from '@shared/lib/cn'
import { input } from './primitives'

// =============================================================================
// SettingsForm primitives — pattern dùng trong Settings tab.
//   <SettingsSection title description>      page-level chunk, separated by divider
//     <SettingsRow label hint>...</SettingsRow>      label-left, control-right
//
// No card wrapper, no uppercase eyebrow header. Linear/Stripe/Plane pattern.
// =============================================================================

export function SettingsSection({
  title, description, children, danger = false,
}: {
  title:        string
  description?: ReactNode
  children:     ReactNode
  danger?:      boolean
}) {
  return (
    <section className={cn(
      'py-6 max-w-2xl',
      danger && 'border border-error/30 rounded-md bg-error-bg/30 px-5',
    )}>
      <header className="mb-5">
        <h2 className={cn('text-base font-semibold', danger ? 'text-error-text' : 'text-text')}>
          {title}
        </h2>
        {description && (
          <p className="text-[13px] text-text-subtle mt-0.5">{description}</p>
        )}
      </header>
      <div className="space-y-5">{children}</div>
    </section>
  )
}

export function SettingsRow({
  label, hint, children,
}: {
  label:    string
  hint?:    ReactNode
  children: ReactNode
}) {
  return (
    <div className="flex items-start gap-8">
      <div className="w-60 shrink-0 pt-1.5">
        <p className="text-[13px] font-medium text-text">{label}</p>
        {hint && <p className="text-xs text-text-subtle mt-1 leading-relaxed">{hint}</p>}
      </div>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}

/**
 * Action row — destructive or one-shot operations. Used in danger zones.
 * Layout: [title + description] ──── [button]
 *
 * Different from SettingsRow which is label/value pair for editable fields.
 * Here the action IS the field, no need for a separate label column.
 */
export function SettingsAction({
  title, description, action,
}: {
  title:        string
  description?: ReactNode
  action:       ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium text-text">{title}</p>
        {description && (
          <p className="text-xs text-text-subtle mt-0.5 leading-relaxed">{description}</p>
        )}
      </div>
      <div className="shrink-0">{action}</div>
    </div>
  )
}
export function SettingsValue({
  children, className,
}: {
  children:   ReactNode
  className?: string
}) {
  return (
    <div className={cn('h-8 flex items-center text-[13px] text-text-muted font-mono min-w-0', className)}>
      {children}
    </div>
  )
}

/** Toggle switch (left-aligned content, right-aligned switch). */
export function SettingsToggle({
  checked, onChange, disabled,
}: {
  checked:   boolean
  onChange:  (v: boolean) => void
  disabled?: boolean
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors cursor-pointer',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        checked ? 'bg-accent' : 'bg-surface-2',
      )}
    >
      <span
        className={cn(
          'inline-block size-4 rounded-full bg-white transition-transform',
          checked ? 'translate-x-[18px]' : 'translate-x-[2px]',
        )}
      />
    </button>
  )
}

/** Textarea with the same shape as `input` but auto-height resize-y. */
export function Textarea({ className, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(input, 'h-auto py-2 resize-y leading-relaxed', className)}
      {...props}
    />
  )
}
