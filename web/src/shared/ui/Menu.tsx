// Menu — lightweight popover anchored to a trigger button. Items are
// either action handlers or links; visual style follows the existing
// surface/hover tokens. Closes on Esc, outside click, or item select.
//
// Not a full headless system — single-level menu, no sub-menus, no
// arrow-key wraparound. Enough for row overflow + table toolbars.
//
// Item kinds (discriminated by `kind` — defaults to `'item'`):
//
//   item     clickable row. Supports `selected` for radio groups
//            (renders a check mark). Supports `danger` for
//            destructive actions (red text). Supports `href` for
//            link-style entries that open in a new tab.
//
//   section  non-interactive header (e.g. "Sắp xếp"). Plain text in
//            text-text-subtle uppercase tracking. Skipped by
//            keyboard nav.
//
//   divider  horizontal rule between groups. No label.

import {
  useEffect, useRef, useState, type ReactNode, type MouseEvent,
} from 'react'
import { Check } from 'lucide-react'
import { cn } from '@shared/lib/cn'


export interface MenuItemAction {
  key:       string
  /** Default kind. Omit for plain items. */
  kind?:     'item'
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
  /** Render a check mark on the right — use for radio groups. Only
   *  one item per group is expected to be selected. */
  selected?: boolean
}

export interface MenuItemSection {
  key:   string
  kind:  'section'
  label: string
}

export interface MenuItemDivider {
  key:  string
  kind: 'divider'
}

export type MenuItem = MenuItemAction | MenuItemSection | MenuItemDivider


interface Props {
  trigger:          ReactNode
  items:            MenuItem[]
  align?:           'start' | 'end'
  className?:       string
  /** Override the trigger button's className (replaces the default
   *  size-7 icon-button shell). Use when the trigger is a full
   *  labeled button rather than an icon. */
  triggerClassName?: string
}

export function Menu({ trigger, items, align = 'end', className, triggerClassName }: Props) {
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
        className={triggerClassName ?? cn(
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
            'bg-surface rounded-md border border-border-soft',
            'py-1',
            align === 'end' ? 'right-0' : 'left-0',
          )}
        >
          {items.map((item) => {
            if (item.kind === 'divider') {
              return (
                <div
                  key={item.key}
                  role="separator"
                  className="my-1 border-t border-border-soft"
                />
              )
            }
            if (item.kind === 'section') {
              return (
                <div
                  key={item.key}
                  className="px-3 pt-2 pb-1 text-xs uppercase tracking-wider text-text-subtle font-medium"
                >
                  {item.label}
                </div>
              )
            }
            return (
              <ActionRow
                key={item.key}
                item={item}
                onClose={() => setOpen(false)}
                stop={stop}
              />
            )
          })}
        </div>
      )}
    </div>
  )
}


function ActionRow({
  item, onClose, stop,
}: {
  item:    MenuItemAction
  onClose: () => void
  stop:    (e: MouseEvent) => void
}) {
  const onClick = (e: MouseEvent) => {
    stop(e)
    if (item.disabled) return
    onClose()
    item.onSelect?.()
  }
  const cls = cn(
    'w-full flex items-center gap-2 px-3 py-1.5 text-sm',
    'text-text-muted hover:text-text hover:bg-hover',
    'transition-colors cursor-pointer text-left',
    item.disabled && 'opacity-40 cursor-not-allowed pointer-events-none',
    item.danger && 'text-error-text hover:text-error-text',
    item.selected && !item.danger && 'text-text font-medium',
  )
  const body = (
    <>
      {item.icon && <span className="size-3.5 inline-flex">{item.icon}</span>}
      <span className="flex-1">{item.label}</span>
      {item.selected && <Check size={12} className="text-text-subtle" />}
    </>
  )
  if (item.href) {
    return (
      <a
        href={item.href}
        target="_blank"
        rel="noopener noreferrer"
        onClick={onClick}
        role="menuitem"
        className={cls}
      >
        {body}
      </a>
    )
  }
  return (
    <button
      type="button"
      onClick={onClick}
      role="menuitem"
      disabled={item.disabled}
      className={cls}
    >
      {body}
    </button>
  )
}
