// SettingsForm primitives — pattern dùng trong Settings tabs.
//
// Hai layer:
//
//   1. Page shell — `SettingsRail` + section header (title/desc/action).
//      Dùng cho /settings (account / sources / tokens).
//
//   2. Per-field — `SettingsField` (label-left, control-right) cho
//      project-detail settings (mỗi project có form chỉnh sửa).
//
// Visual: solid `bg-surface`, `border-soft` dividers, no card chrome.
// No uppercase eyebrow headers — Linear/Stripe/Plane pattern.

import { type ReactNode } from 'react'
import { useNavigate, useSearch } from '@tanstack/react-router'
import { type LucideIcon } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import { input } from './primitives'

// ════════════════════════════════════════════════════════════════════════════
// Layer 1 — page shell
// ════════════════════════════════════════════════════════════════════════════

export interface SettingsTab {
  id:    string
  label: string
  icon?: LucideIcon
  hint?: string
}

/** Vertical rail nav. Active row paints `surface-2 + accent icon`. */
export function SettingsRail({
  tabs, active, onChange,
}: {
  tabs:     SettingsTab[]
  active:   string
  onChange: (id: string) => void
}) {
  return (
    <nav className="flex sm:flex-col gap-0.5 w-full sm:w-[220px] shrink-0 overflow-x-auto sm:overflow-visible">
      {tabs.map((t) => {
        const Icon = t.icon
        const isActive = t.id === active
        return (
          <button
            key={t.id}
            onClick={() => onChange(t.id)}
            className={cn(
              'flex items-center gap-2.5 h-9 px-3 rounded-sm text-sm cursor-pointer',
              'transition-colors text-left w-full whitespace-nowrap',
              isActive
                ? 'bg-surface-2 text-text font-medium'
                : 'text-text-muted hover:bg-hover hover:text-text',
            )}
            title={t.hint}
          >
            {Icon && (
              <Icon
                size={14}
                className={cn(
                  'shrink-0',
                  isActive ? 'text-accent-text' : 'text-text-subtle',
                )}
              />
            )}
            <span className="truncate">{t.label}</span>
          </button>
        )
      })}
    </nav>
  )
}

/** Section header inline with primary action. */
export function SettingsSection({
  title, description, action, children, danger = false,
}: {
  title:        string
  description?: ReactNode
  /** Optional control rendered inline with the section header. */
  action?:      ReactNode
  children:     ReactNode
  danger?:      boolean
}) {
  return (
    <section className={cn(
      'pb-8',
      danger && 'mt-8 pt-6 border-t border-error/30',
    )}>
      <header className="mb-4">
        <div className="flex items-center justify-between gap-4 mb-1.5">
          <h2 className={cn(
            'text-base font-semibold tracking-tight',
            danger ? 'text-error-text' : 'text-text',
          )}>
            {title}
          </h2>
          {action && <div className="shrink-0">{action}</div>}
        </div>
        {description && (
          <p className="text-sm text-text-subtle leading-relaxed max-w-prose">
            {description}
          </p>
        )}
      </header>
      <div className="space-y-3">{children}</div>
    </section>
  )
}

/** Thin horizontal divider between settings sections (kept for callers
 *  that still stack sections vertically). */
export function SettingsDivider() {
  return <hr className="border-0 h-px bg-border-soft max-w-2xl" />
}

/** Action row — destructive or one-shot operations.
 *  Layout: [title + description] ──── [button] */
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
        <p className="text-sm font-medium text-text">{title}</p>
        {description && (
          <p className="text-xs text-text-subtle mt-0.5 leading-relaxed">{description}</p>
        )}
      </div>
      <div className="shrink-0">{action}</div>
    </div>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// Layer 2 — per-field (label/control pair)
// ════════════════════════════════════════════════════════════════════════════

/** Label-left, control-right form row. Used for project settings forms. */
export function SettingsField({
  label, hint, children,
}: {
  label:    string
  hint?:    ReactNode
  children: ReactNode
}) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-start gap-2 sm:gap-8">
      <div className="sm:w-60 sm:shrink-0 sm:pt-1.5">
        <p className="text-sm font-medium text-text">{label}</p>
        {hint && <p className="text-xs text-text-subtle mt-1 leading-relaxed">{hint}</p>}
      </div>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}

/** Static value display inside SettingsField. */
export function SettingsValue({
  children, className,
}: {
  children:   ReactNode
  className?: string
}) {
  return (
    <div className={cn('h-8 flex items-center text-sm text-text-muted font-mono min-w-0', className)}>
      {children}
    </div>
  )
}

/** Toggle switch. */
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

/** Textarea with the same shape as `input` but auto-height. */
export function Textarea({ className, ...props }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(input, 'h-auto py-2 resize-y leading-relaxed', className)}
      {...props}
    />
  )
}

// ════════════════════════════════════════════════════════════════════════════
// Layer 3 — list shell (Sources / Tokens style)
// ════════════════════════════════════════════════════════════════════════════

/** Shell wrapping list rows. divide-y for separator, rounded for grouping. */
export function SettingsList({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <ul className={cn(
      'rounded-md bg-surface divide-y divide-border-soft overflow-hidden',
      className,
    )}>
      {children}
    </ul>
  )
}

/** Single list row — flex layout, caller provides content. */
export function SettingsListRow({
  children, className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <li className={cn('px-4 py-3 flex items-center gap-3', className)}>
      {children}
    </li>
  )
}

// ════════════════════════════════════════════════════════════════════════════
// URL-state-bound tab hook
// ════════════════════════════════════════════════════════════════════════════

/** Reads `?section=` and provides a setter preserving other search params. */
export function useSettingsTab(defaultTab: string) {
  const search = useSearch({ strict: false }) as { section?: string }
  const nav    = useNavigate()
  const active = search.section ?? defaultTab
  const setActive = (id: string) =>
    nav({ search: ((s: Record<string, unknown>) => ({ ...s, section: id })) as never })
  return [active, setActive] as const
}
