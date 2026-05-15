// TapZones — invisible 3-zone overlay covering the reader body.
//
//   ┌─────────────────────────────────────┐
//   │ ZONE_PREV │  ZONE_PEEK  │ ZONE_NEXT │
//   │   35%     │     30%     │   35%     │
//   └─────────────────────────────────────┘
//
// Justification:
//
// - DA portrait-locks; user can't bring up the URL bar. Tap zones
//   make the page itself the primary control surface.
// - 35/30/35 (vs an even 33/33/33) matches truyendrive/mangadex
//   convention; gives the centre peek zone enough size to land on
//   without the user accidentally hitting prev/next when reaching
//   for the page count.
// - RTL flips prev/next ONLY. The middle peek zone never swaps.
// - In TTB mode the horizontal zones still work for page-step
//   inside the strip (scroll one viewport at a time). Most users
//   prefer continuous scroll, but step is useful for bookmarking.
//
// Discrimination tap vs scroll: pointer down → up within 10px and
// 400ms is a tap. Anything else is treated as a scroll/drag.

import { useCallback, useRef } from 'react'

import { cn } from '@shared/lib/cn'


export interface TapZoneCallbacks {
  onPrev:       () => void
  onNext:       () => void
  onTogglePeek: () => void
}


interface Props extends TapZoneCallbacks {
  /** When true, left zone becomes "next" and right becomes "prev". */
  rtl?:     boolean
  /** Pause zone reaction (e.g. modal/sheet open). Pointer events
   *  still pass through to children for scroll. */
  disabled?: boolean
}


const TAP_MOVE_TOLERANCE = 10
const TAP_TIMEOUT_MS = 400


export function TapZones({
  onPrev, onNext, onTogglePeek,
  rtl = false, disabled = false,
}: Props) {
  const startRef = useRef<{ x: number; y: number; t: number } | null>(null)

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (disabled) return
    if (e.pointerType === 'mouse' && e.button !== 0) return
    startRef.current = { x: e.clientX, y: e.clientY, t: performance.now() }
  }, [disabled])

  const onPointerUp = useCallback((e: React.PointerEvent) => {
    if (disabled) return
    const start = startRef.current
    startRef.current = null
    if (!start) return

    const dx = Math.abs(e.clientX - start.x)
    const dy = Math.abs(e.clientY - start.y)
    const dt = performance.now() - start.t
    if (dx > TAP_MOVE_TOLERANCE || dy > TAP_MOVE_TOLERANCE) return
    if (dt > TAP_TIMEOUT_MS) return

    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const x = e.clientX - rect.left
    const w = rect.width
    if (w <= 0) return

    if (x < w * 0.35) {
      if (rtl) onNext()
      else onPrev()
    } else if (x > w * 0.65) {
      if (rtl) onPrev()
      else onNext()
    } else {
      onTogglePeek()
    }
  }, [disabled, rtl, onPrev, onNext, onTogglePeek])

  return (
    <div
      className={cn(
        // Overlay sits above the page body but BELOW any sheets /
        // modals (those use z-30+). pointer-events-auto only on the
        // overlay itself; children of the page body remain
        // interactive via their own z-index when needed.
        'absolute inset-0 z-10',
        disabled && 'pointer-events-none',
      )}
      onPointerDown={onPointerDown}
      onPointerUp={onPointerUp}
      aria-hidden
    />
  )
}
