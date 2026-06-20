import { useEffect, type ReactNode } from 'react'
import { X } from 'lucide-react'
import { cn } from '@shared/lib/cn'

// =============================================================================
// Modal — three-zone layout (sticky header, scrollable body, sticky footer).
//
// Pro pattern: header + footer pinned, only body scrolls. Primary action
// stays one click away regardless of scroll depth, and the title/close
// stay visible so users don't lose context in long forms.
//
// Footer can be a single slot (`footer`) for simple right-aligned actions,
// or split via `footerLeft` (live context: "12 ảnh · 24 MB") + `footer`
// (action buttons). When the entire footer should be replaced — e.g. by a
// progress strip during async submission — pass `footerCustom` and skip
// both `footer` and `footerLeft`.
// =============================================================================

// Esc key routing: only the top-most open Modal closes on Escape.
// Without this, nested modals (e.g. Confirm opened from inside another Modal)
// would all close on a single Esc press — which can re-open the parent's
// guard-confirm in a loop.
const escStack: Array<() => void> = []

if (typeof document !== 'undefined') {
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || escStack.length === 0) return
    e.stopPropagation()
    escStack[escStack.length - 1]!()
  })
}

interface Props {
  open:     boolean
  onClose:  () => void
  title:    string
  size?:    'sm' | 'md' | 'lg'
  children: ReactNode
  /** Right-aligned action slot in the footer. */
  footer?:      ReactNode
  /** Optional left-aligned context line ("12 ảnh · 24 MB · Ch.5"). */
  footerLeft?:  ReactNode
  /** Replace the entire footer with a custom node (e.g. async progress strip). */
  footerCustom?: ReactNode
  /**
   * Stacking layer. Default `base` (z-50). Use `top` (z-70) for modals that
   * may open from inside another modal — e.g. the global Confirm dialog —
   * so they always render above the parent and above toast notifications.
   */
  layer?:   'base' | 'top'
}

const SIZE = {
  sm: 'max-w-md',
  md: 'max-w-2xl',
  lg: 'max-w-4xl',
}

const LAYER = {
  base: 'z-50',
  top:  'z-[70]',
}

export function Modal({
  open, onClose, title, size = 'md', children,
  footer, footerLeft, footerCustom, layer = 'base',
}: Props) {
  useEffect(() => {
    if (!open) return
    escStack.push(onClose)
    return () => {
      const i = escStack.lastIndexOf(onClose)
      if (i >= 0) escStack.splice(i, 1)
    }
  }, [open, onClose])

  if (!open) return null

  const hasFooter = footerCustom !== undefined || footer !== undefined || footerLeft !== undefined

  return (
    <div
      className={cn(
        'fixed inset-0 flex items-center justify-center bg-black/60',
        'p-4',
        'pt-[max(1rem,var(--sait))] pb-[max(1rem,var(--saib))]',
        'pl-[max(1rem,var(--sail))] pr-[max(1rem,var(--sair))]',
        LAYER[layer],
      )}
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className={cn(
          'w-full bg-surface text-text rounded-md border border-border-soft',
          'flex flex-col max-h-[88vh] overflow-hidden',
          SIZE[size],
        )}
      >
        <header className="flex items-center justify-between px-5 h-[52px] border-b border-border-soft shrink-0">
          <h2 className="text-base font-semibold text-text tracking-tight truncate">{title}</h2>
          <button
            onClick={onClose}
            className="size-8 rounded-sm flex items-center justify-center text-text-subtle hover:text-text hover:bg-hover transition-colors cursor-pointer"
            aria-label="Đóng"
          >
            <X size={15} />
          </button>
        </header>

        <div className="flex-1 overflow-auto overscroll-contain">
          {children}
        </div>

        {hasFooter && (
          footerCustom !== undefined ? (
            <footer className="border-t border-border-soft bg-bg/40 shrink-0">
              {footerCustom}
            </footer>
          ) : (
            <footer className="flex items-center gap-3 px-5 py-3 border-t border-border-soft bg-bg/40 shrink-0">
              <div className="flex-1 min-w-0 text-xs text-text-subtle truncate">
                {footerLeft}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {footer}
              </div>
            </footer>
          )
        )}
      </div>
    </div>
  )
}
