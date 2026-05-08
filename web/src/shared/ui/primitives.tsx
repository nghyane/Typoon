import { cn } from '@shared/lib/cn'

// =============================================================================
// Form input — token-driven, no border drift between routes.
// Use `<input className={input}>` for plain text inputs and pair with <Field/>
// for label + hint layout.
// =============================================================================

export const input =
  'h-8 w-full px-3 rounded-sm bg-surface-2 border border-transparent ' +
  'text-sm text-text placeholder:text-text-subtle ' +
  'hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none ' +
  'transition-colors'

export const label = 'block text-xs font-medium text-text-muted mb-1.5'

export function Field({
  label: lbl, hint, children,
}: {
  label:    string
  hint?:    string
  children: React.ReactNode
}) {
  return (
    <div>
      <label className={label}>{lbl}</label>
      {children}
      {hint && <p className="text-xs text-text-subtle mt-1">{hint}</p>}
    </div>
  )
}

// ── badges ──────────────────────────────────────────────────────────────────

export type BadgeTone = 'success' | 'info' | 'warning' | 'neutral' | 'error'

const BADGE_BASE =
  'inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full text-[11px] font-medium'

const BADGE_TONE: Record<BadgeTone, string> = {
  success: 'bg-success-bg text-success-text',
  info:    'bg-info-bg text-info-text',
  warning: 'bg-warning-bg text-warning-text',
  neutral: 'bg-surface-2 text-text-muted',
  error:   'bg-error-bg text-error-text',
}

const DOT_TONE: Record<BadgeTone, string> = {
  success: 'bg-success',
  info:    'bg-info',
  warning: 'bg-warning',
  neutral: 'bg-text-subtle',
  error:   'bg-error',
}

export function Badge({
  tone, dot = true, children, className,
}: {
  tone:     BadgeTone
  dot?:     boolean
  children: React.ReactNode
  className?: string
}) {
  return (
    <span className={cn(BADGE_BASE, BADGE_TONE[tone], className)}>
      {dot && <span className={cn('size-1.5 rounded-full flex-none', DOT_TONE[tone])} />}
      {children}
    </span>
  )
}

// ── card / list shells ─────────────────────────────────────────────────────

export const card = 'bg-surface rounded-md'
export const list = 'bg-surface rounded-md overflow-hidden'

// ── spinner ────────────────────────────────────────────────────────────────

export function Spinner({ size = 14, className }: { size?: number; className?: string }) {
  return (
    <span
      className={cn('inline-block ts-spinner-circle', className)}
      style={{ width: size, height: size }}
      aria-label="Đang tải"
      role="status"
    />
  )
}

// ── monogram avatar (hash-based bg color) ──────────────────────────────────

const AVATAR_PALETTE = [
  '#4F88E6', '#23A55A', '#F0B232', '#F47B67',
  '#8B5CF6', '#EC4899', '#0EA5E9', '#10B981',
] as const

function hashColor(seed: string): string {
  let h = 0
  for (let i = 0; i < seed.length; i++) h = (h << 5) - h + seed.charCodeAt(i)
  return AVATAR_PALETTE[Math.abs(h) % AVATAR_PALETTE.length]!
}

export function Monogram({
  name, size = 28, className,
}: {
  name: string
  size?: number
  className?: string
}) {
  const initial = (name.trim().charAt(0) || '?').toUpperCase()
  return (
    <span
      className={cn(
        'inline-grid place-items-center rounded-full text-white font-semibold flex-none',
        className,
      )}
      style={{
        width: size,
        height: size,
        background: hashColor(name),
        fontSize: Math.round(size * 0.4),
      }}
    >
      {initial}
    </span>
  )
}

// ── KPI tile ───────────────────────────────────────────────────────────────

export function Stat({
  label: lbl, value, sub, valueClass,
}: {
  label:      string
  value:      React.ReactNode
  sub?:       React.ReactNode
  valueClass?: string
}) {
  return (
    <div className={cn(card, 'p-4')}>
      <p className="text-[11px] uppercase tracking-wider text-text-subtle font-medium">
        {lbl}
      </p>
      <p className={cn('text-2xl font-semibold leading-tight mt-1.5', valueClass ?? 'text-text')}>
        {value}
      </p>
      <p className="text-xs text-text-subtle mt-1.5 min-h-4">{sub}</p>
    </div>
  )
}
