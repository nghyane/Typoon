// HoverEdges — desktop affordance: hovering the left/right 8% of
// the reader surfaces a chevron in the centre of that edge,
// clickable to navigate prev/next chapter (or page in pager mode).
//
// Desktop only (`hidden sm:block`) — mobile uses TapZones, where
// hover doesn't make sense. RTL is purely visual: left edge always
// drives "prev" semantically (the tap-zone wrapper handles
// direction swap).

import { ChevronLeft, ChevronRight } from 'lucide-react'

import { cn } from '@shared/lib/cn'


interface Props {
  onLeft:  () => void
  onRight: () => void
  /** When true the chevrons swap so the "back arrow" visually
   *  points toward the next chapter (the RTL reading direction).
   *  Useful affordance for manga readers who expect the right side
   *  to flow forward. */
  rtl?: boolean
  /** Disable when a sheet is open so clicking through doesn't jump
   *  chapters by mistake. */
  disabled?: boolean
}


export function HoverEdges({
  onLeft, onRight, rtl = false, disabled = false,
}: Props) {
  const LeftIcon  = rtl ? ChevronRight : ChevronLeft
  const RightIcon = rtl ? ChevronLeft  : ChevronRight

  return (
    <div
      className={cn(
        'hidden sm:block absolute inset-y-0 left-0 right-0 z-10',
        'pointer-events-none',   // children opt-in
        disabled && 'opacity-0',
      )}
      aria-hidden
    >
      <button
        onClick={onLeft}
        disabled={disabled}
        className={cn(
          'pointer-events-auto absolute inset-y-0 left-0 w-[8%]',
          'flex items-center justify-center',
          'text-transparent hover:text-text-muted hover:bg-hover',
          'transition-all duration-150 cursor-pointer',
          'disabled:cursor-default disabled:hover:bg-transparent',
          'group',
        )}
        aria-label="Trang/chương trước"
      >
        <LeftIcon size={32} className="opacity-0 group-hover:opacity-100" />
      </button>

      <button
        onClick={onRight}
        disabled={disabled}
        className={cn(
          'pointer-events-auto absolute inset-y-0 right-0 w-[8%]',
          'flex items-center justify-center',
          'text-transparent hover:text-text-muted hover:bg-hover',
          'transition-all duration-150 cursor-pointer',
          'disabled:cursor-default disabled:hover:bg-transparent',
          'group',
        )}
        aria-label="Trang/chương sau"
      >
        <RightIcon size={32} className="opacity-0 group-hover:opacity-100" />
      </button>
    </div>
  )
}
