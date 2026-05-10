import type { ReactNode } from 'react'
import { cn } from '@shared/lib/cn'

// Input + Field styles copied from web/src/shared/ui/primitives.tsx.
// Identical class string so popup form rows look pixel-identical to
// SPA Settings rows when placed next to each other.

export const input =
  'h-8 w-full px-3 rounded-sm bg-surface-2 border border-transparent ' +
  'text-sm text-text placeholder:text-text-subtle ' +
  'hover:bg-hover focus:border-accent focus:bg-surface-2 focus:outline-none ' +
  'transition-colors'

export const labelClass = 'block text-xs font-medium text-text-muted mb-1.5'

export function Field({
  label, hint, children,
}: {
  label:    string
  hint?:    ReactNode
  children: ReactNode
}) {
  return (
    <div>
      <label className={labelClass}>{label}</label>
      {children}
      {hint && <p className="text-xs text-text-subtle mt-1">{hint}</p>}
    </div>
  )
}

// Spinner — same `ts-spinner-circle` keyframe lives in
// `entrypoints/popup/style.css` (we copied the rule from web/index.css
// rather than the SPA's own primitives module).
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
