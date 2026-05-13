import { cn } from '@shared/lib/cn'

// =============================================================================
// Language code chip picker — dùng cho source/target lang trong settings,
// tạo project, … Active chip dùng accent fill.
// =============================================================================

export interface LangOption { code: string; label: string }

interface Props {
  value:    string
  onChange: (v: string) => void
  options:  readonly LangOption[]
  disabled?: boolean
}

export function LangPicker({ value, onChange, options, disabled }: Props) {
  return (
    <div className="inline-flex flex-wrap gap-2">
      {options.map((l) => {
        const active = value === l.code
        return (
          <button
            key={l.code}
            type="button"
            onClick={() => onChange(l.code)}
            disabled={disabled}
            className={cn(
              'h-8 px-3 rounded-sm text-xs font-medium cursor-pointer transition-colors',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              active
                ? 'bg-accent text-accent-fg'
                : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
            )}
          >
            {l.label}
          </button>
        )
      })}
    </div>
  )
}
