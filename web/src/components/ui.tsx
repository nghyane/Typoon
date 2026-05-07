import { cn } from '../lib/cn'

// Single source of truth for button styling — every route was duplicating
// these exact classes inline.

export const btn = {
  primary:
    'inline-flex items-center justify-center gap-1.5 h-9 px-4 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all cursor-pointer',
  secondary:
    'inline-flex items-center justify-center gap-1.5 h-9 px-3.5 rounded-lg border border-zinc-200 text-sm text-zinc-600 bg-white hover:bg-zinc-50 hover:border-zinc-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer',
  danger:
    'inline-flex items-center justify-center gap-1.5 h-9 px-4 rounded-lg bg-red-600 text-white text-sm font-medium hover:bg-red-700 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all cursor-pointer',
  iconBox:
    'size-9 rounded-lg flex items-center justify-center border border-zinc-200 bg-white text-zinc-500 hover:bg-zinc-100 hover:border-zinc-300 transition-colors cursor-pointer',
  ghost:
    'size-7 rounded-md flex items-center justify-center text-zinc-400 hover:text-zinc-700 hover:bg-zinc-100 transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed',
} as const

export const input =
  'h-9 w-full px-3 rounded-lg border border-zinc-200 bg-white text-sm text-zinc-900 placeholder:text-zinc-400 hover:border-zinc-300 focus:border-zinc-400 focus:ring-2 focus:ring-zinc-100 focus:outline-none transition-colors'

export const label = 'block text-xs font-medium text-zinc-600 mb-1.5'

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
      {hint && <p className="text-xs text-zinc-400 mt-1">{hint}</p>}
    </div>
  )
}

export function Spinner({ size = 14, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      className={cn('animate-spin', className)}
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" fill="none" opacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" fill="none" />
    </svg>
  )
}
