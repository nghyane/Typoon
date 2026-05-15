// BottomSheet — mobile-targeted slide-up panel. Used when a
// dropdown's content is too long for a popover anchored to a
// chip (chapter lists with 90+ entries, complex multi-section
// forms). Pairs with `<Popover>` via a higher-level component
// that switches between them based on viewport width.
//
// Visual rules:
//   - Drag handle at top so the sheet reads as a sheet, not a
//     modal that happens to slide from the bottom.
//   - Backdrop is subtle (no blur, low opacity). The point of a
//     bottom-sheet is the user is still in the page — the dim
//     just kills behind-page interaction without hiding context.
//   - Rounded top corners only. Bottom flush with the viewport's
//     safe-area inset.
//   - Max height 80dvh so the user can always see a slice of the
//     page above and remembers they're not in a different screen.
//
// Closing rules: Esc-stack (top-most only), backdrop click, and
// the caller's onClose. No swipe-down-to-dismiss yet — keep that
// for a v2; the visible Close button at top-right covers the
// affordance.

import { useEffect, type ReactNode } from 'react'
import { X } from 'lucide-react'

import { cn } from '@shared/lib/cn'


const escStack: Array<() => void> = []

if (typeof document !== 'undefined') {
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || escStack.length === 0) return
    e.stopPropagation()
    escStack[escStack.length - 1]!()
  })
}


export interface BottomSheetProps {
  open:     boolean
  onClose:  () => void
  /** Header line. Short string only (the sheet is title-first
   *  visually, not a card with arbitrary header content). */
  title?:   string
  /** Optional sticky footer (e.g. "Đặt làm mặc định" checkbox).
   *  Use sparingly — most sheets won't need one. */
  footer?:  ReactNode
  children: ReactNode
}


export function BottomSheet({
  open, onClose, title, footer, children,
}: BottomSheetProps) {
  useEffect(() => {
    if (!open) return
    escStack.push(onClose)
    return () => {
      const i = escStack.lastIndexOf(onClose)
      if (i >= 0) escStack.splice(i, 1)
    }
  }, [open, onClose])

  return (
    <>
      <div
        className={cn(
          'fixed inset-0 z-40 bg-black/40',
          'transition-opacity duration-200',
          open ? 'opacity-100' : 'opacity-0 pointer-events-none',
        )}
        onClick={onClose}
        aria-hidden
      />

      <aside
        role="dialog"
        aria-modal
        aria-label={title}
        className={cn(
          'fixed inset-x-0 bottom-0 z-50',
          'max-h-[80dvh]',
          'flex flex-col',
          'bg-surface text-text',
          'rounded-t-xl shadow-2xl',
          'pb-[var(--saib)]',
          'transition-transform duration-200 ease-out',
          !open && 'translate-y-full',
        )}
      >
        {/* Drag handle — read-only visual; tap-to-dismiss handled
            by backdrop click. Adding gesture drag in v2. */}
        <div className="pt-2 pb-1 flex justify-center">
          <span className="block w-9 h-1 rounded-full bg-divider" />
        </div>

        {title && (
          <header className="flex items-center justify-between px-4 pb-2 shrink-0">
            <h2 className="text-sm font-semibold tracking-tight">
              {title}
            </h2>
            <button
              onClick={onClose}
              className={cn(
                'size-8 -mr-1 rounded-md',
                'flex items-center justify-center',
                'text-text-subtle hover:text-text hover:bg-hover',
                'transition-colors duration-150 cursor-pointer',
              )}
              aria-label="Đóng"
            >
              <X size={16} />
            </button>
          </header>
        )}

        <div className="flex-1 overflow-y-auto overscroll-contain">
          {children}
        </div>

        {footer && (
          <footer className="shrink-0 border-t border-border-soft px-4 py-3">
            {footer}
          </footer>
        )}
      </aside>
    </>
  )
}
