// Popover — anchored panel attached to a trigger element. Positions
// itself relative to the trigger's bounding rect; recalculates on
// scroll / resize so it stays glued during page motion.
//
// Closing rules (Esc-stack aware, mirrors `Modal`):
//   - Click outside both trigger and panel.
//   - Esc, only when this is the top-most open Popover.
//   - The caller explicitly setting `open` to false.
//
// Pro UI rules baked in:
//   - No backdrop / dim. The whole point of "anchored vs modal" is
//     the page area stays interactive and readable.
//   - Auto flip vertically: if there isn't enough room below the
//     trigger, the panel renders ABOVE it. This is what makes the
//     "drop down from top bar" feel correct even on short
//     viewports.
//   - Auto flip horizontally: align right edge of panel to right
//     edge of trigger by default; switch to left edge if that would
//     overflow viewport.
//   - Fixed positioning so the panel doesn't get clipped by
//     overflow-hidden ancestors (the reader has a few).
//
// Caller wires `anchorRef` to whichever button opens this popover.
// Multiple popovers are mutually exclusive at the call site
// (`useReaderUiState` enforces it), so this primitive doesn't try
// to coordinate sibling popovers.

import {
  useEffect, useLayoutEffect, useRef, useState,
  type ReactNode, type RefObject,
} from 'react'

import { cn } from '@shared/lib/cn'


// Esc-stack: only the top-most open Popover responds to Escape.
// Mirrors the pattern in `shared/ui/Modal.tsx` so a Popover opened
// inside a Modal doesn't double-close on a single Esc press.
const escStack: Array<() => void> = []

if (typeof document !== 'undefined') {
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || escStack.length === 0) return
    e.stopPropagation()
    escStack[escStack.length - 1]!()
  })
}


export interface PopoverProps {
  open:       boolean
  onClose:    () => void
  /** The button (or any element) the popover anchors to. */
  anchorRef:  RefObject<HTMLElement | null>
  /** Horizontal alignment of the panel's edge against the
   *  trigger's edge. `end` aligns right edge to right edge (the
   *  common pattern for top-bar chips); `start` aligns left to
   *  left. `center` centres the panel under the trigger. Flips
   *  automatically when the chosen alignment would overflow the
   *  viewport. */
  align?:     'start' | 'end' | 'center'
  /** Px gap between the trigger and the panel. Default 6. */
  offset?:    number
  /** Min / max width of the panel content. The panel will use the
   *  natural width of its content within these bounds. */
  minWidth?:  number
  maxWidth?:  number
  className?: string
  children:   ReactNode
}


export function Popover({
  open, onClose, anchorRef,
  align = 'end', offset = 6,
  minWidth = 240, maxWidth = 480,
  className, children,
}: PopoverProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<{
    top: number; left: number; placement: 'below' | 'above'
  } | null>(null)

  // Compute placement against the trigger's rect. Done in a layout
  // effect so the first paint already has correct coordinates (no
  // jitter from a one-frame [0,0] -> [actual] swap).
  useLayoutEffect(() => {
    if (!open) { setPos(null); return }
    const compute = () => {
      const trigger = anchorRef.current
      const panel   = panelRef.current
      if (!trigger || !panel) return

      const tr = trigger.getBoundingClientRect()
      const pr = panel.getBoundingClientRect()
      const vw = window.innerWidth
      const vh = window.innerHeight

      // Vertical: prefer below, flip above if overflow.
      const below = tr.bottom + offset
      const above = tr.top - offset - pr.height
      const placement: 'below' | 'above' =
        below + pr.height <= vh - 8 ? 'below'
        : above >= 8 ? 'above'
        : 'below'   // both overflow: fall back to below (clip handled by max-height)
      const top = placement === 'below' ? below : above

      // Horizontal: align trigger edge, flip if overflow.
      let left: number
      if (align === 'end') {
        left = tr.right - pr.width
        if (left < 8) left = tr.left
      } else if (align === 'start') {
        left = tr.left
        if (left + pr.width > vw - 8) left = tr.right - pr.width
      } else {
        left = tr.left + (tr.width - pr.width) / 2
      }
      // Clamp to viewport with 8px margin so we never disappear
      // off-screen on a viewport resize.
      left = Math.max(8, Math.min(left, vw - pr.width - 8))

      setPos({ top, left, placement })
    }

    // First measure: the panel just mounted; measure its natural
    // size after the browser has laid it out.
    requestAnimationFrame(compute)

    const onResize = () => requestAnimationFrame(compute)
    window.addEventListener('resize', onResize)
    window.addEventListener('scroll', onResize, true)
    return () => {
      window.removeEventListener('resize', onResize)
      window.removeEventListener('scroll', onResize, true)
    }
  }, [open, anchorRef, align, offset])

  // Click outside both trigger and panel → close.
  useEffect(() => {
    if (!open) return
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node
      if (panelRef.current?.contains(t)) return
      if (anchorRef.current?.contains(t)) return
      onClose()
    }
    // pointerdown not click — fires before focus change, before
    // a button inside the popover would steal the event.
    document.addEventListener('pointerdown', onDown)
    return () => document.removeEventListener('pointerdown', onDown)
  }, [open, anchorRef, onClose])

  // Esc-stack registration.
  useEffect(() => {
    if (!open) return
    escStack.push(onClose)
    return () => {
      const i = escStack.lastIndexOf(onClose)
      if (i >= 0) escStack.splice(i, 1)
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      ref={panelRef}
      role="dialog"
      style={{
        position: 'fixed',
        top:      pos?.top ?? -9999,
        left:     pos?.left ?? -9999,
        minWidth,
        maxWidth,
        // Hide until first measure to avoid the [0,0] flash.
        visibility: pos ? 'visible' : 'hidden',
      }}
      className={cn(
        'z-50',
        'bg-surface text-text rounded-md border border-border-soft',
        'shadow-xl',
        'flex flex-col overflow-hidden',
        // Defensive max-height so a long list doesn't blow past
        // the viewport when there isn't space to flip.
        'max-h-[calc(100dvh-1rem)]',
        className,
      )}
    >
      {children}
    </div>
  )
}
