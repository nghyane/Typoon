// MenuShell — the picker-shell pattern used by the reader's
// chapter/source pickers. Auto-selects between a desktop Popover
// (anchored to a trigger element) and a mobile BottomSheet
// (full-width slide-up).
//
// Why two pickers don't each re-implement this:
//   - Desktop = anchored = preserves page context.
//   - Mobile  = bottom-sheet = thumb-reachable, can host taller
//                              content (search + long list).
//   - The crossover point is a single Tailwind breakpoint, so
//     "which surface" should never be a decision the caller
//     makes — it's a media-query truth.
//
// Caller passes:
//   - The trigger ref (for anchored mode).
//   - A title (used by BottomSheet header; ignored by Popover
//     unless the caller renders one internally).
//   - Optional sticky footer (e.g. "đặt làm mặc định" toggle).
//   - The body children.

import type { ReactNode, RefObject } from 'react'

import { useIsDesktop } from '@shared/lib/useMediaQuery'
import { Popover } from './Popover'
import { BottomSheet } from './BottomSheet'


export interface MenuShellProps {
  open:       boolean
  onClose:    () => void
  anchorRef:  RefObject<HTMLElement | null>
  /** Used as the BottomSheet header on mobile; passed through to
   *  ARIA on desktop. */
  title:      string
  footer?:    ReactNode
  /** Desktop popover sizing. Mobile sheet ignores these and uses
   *  its own viewport-based height. */
  minWidth?:  number
  maxWidth?:  number
  align?:     'start' | 'end' | 'center'
  children:   ReactNode
}


export function MenuShell({
  open, onClose, anchorRef, title, footer,
  minWidth, maxWidth, align,
  children,
}: MenuShellProps) {
  const isDesktop = useIsDesktop()

  if (isDesktop) {
    return (
      <Popover
        open={open}
        onClose={onClose}
        anchorRef={anchorRef}
        align={align}
        minWidth={minWidth}
        maxWidth={maxWidth}
      >
        <div className="flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto overscroll-contain">
            {children}
          </div>
          {footer && (
            <div className="shrink-0 border-t border-border-soft">
              {footer}
            </div>
          )}
        </div>
      </Popover>
    )
  }

  return (
    <BottomSheet open={open} onClose={onClose} title={title} footer={footer}>
      {children}
    </BottomSheet>
  )
}
