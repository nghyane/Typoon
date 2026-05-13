// Menu — lightweight popover anchored to a trigger button. Items are
// either action handlers or links; visual style follows the existing
// surface/hover tokens. Closes on Esc, outside click, or item select.
//
// Not a full headless system — single-level menu, no sub-menus, no
// arrow-key wraparound. Enough for row overflow + table toolbars.

import {
  useEffect, useRef, useState, type ReactNode, type MouseEvent,
} from 'react'
import { cn } from '@shared/lib/cn'

export interface MenuItem {
  key:       string
  label:     string
  icon?:     ReactNode
  /** Mutually exclusive with `to`. */
  onSelect?: () => void
  /** External URL — opens in new tab. */
  href?:     string
  /** Tooltip + visual greying. */
  disabled?: boolean
  /** Render as destructive (red text). */
  danger?:   boolean
}

interface Props {
  trigger:   ReactNode
  items:     MenuItem[]
  align?:    'start' | 'end'
  className?: string
}

export function Menu({ trigger, items, align = 'end', className }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Close on outside click + Escape. Mount only when open so we don't
  // keep document listeners alive for every menu on the page.
  useEffect(() => {
    if (!open) return
    const onDocClick = (e: globalThis.MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  // Prevent the row click handler from firing when interacting with
  // the menu — callers usually mount this inside a clickable row.
  const stop = (e: MouseEvent) => e.stopPropagation()

  return (
    <div ref={ref} className={cn('relative inline-flex', className)} onClick={stop}>
      <button
        type="button"
        onClick={(e) => { stop(e); setOpen((o) => !o) }}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(
          'inline-flex items-center justify-center size-7 rounded-sm',
          'text-text-subtle hover:text-text hover:bg-hover',
          'transition-colors cursor-pointer',
          open && 'bg-hover text-text',
        )}
      >
        {trigger}
      </button>

      {open && (
        <div
          role="menu"
          className={cn(
            'absolute top-full mt-1 z-30 min-w-[180px]',
            'bg-surface rounded-md shadow-[0_8px_24px_rgb(0,0,0,0.35)] border border-border-soft',
            'py-1',
            align === 'end' ? 'right-0' : 'left-0',
          )}
        >
          {items.map((item) => {
            const onClick = (e: MouseEvent) => {
              stop(e)
              if (item.disabled) return
              setOpen(false)
              item.onSelect?.()
            }
            const cls = cn(
              'w-full flex items-center gap-2 px-3 py-1.5 text-sm',
              'text-text-muted hover:text-text hover:bg-hover',
              'transition-colors cursor-pointer text-left',
              item.disabled && 'opacity-40 cursor-not-allowed pointer-events-none',
              item.danger && 'text-error-text hover:text-error-text',
            )
            if (item.href) {
              return (
                <a
                  key={item.key}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={onClick}
                  role="menuitem"
                  className={cls}
                >
                  {item.icon && <span className="size-3.5 inline-flex">{item.icon}</span>}
                  {item.label}
                </a>
              )
            }
            return (
              <button
                key={item.key}
                type="button"
                onClick={onClick}
                role="menuitem"
                disabled={item.disabled}
                className={cls}
              >
                {item.icon && <span className="size-3.5 inline-flex">{item.icon}</span>}
                {item.label}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
