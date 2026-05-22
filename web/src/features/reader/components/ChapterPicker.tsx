// ChapterPicker — popover (desktop) / bottom sheet (mobile) listing
// every chapter in the work. Auto-scrolls the current chapter into
// view on open. Search filters by chapter number / label.

import {
  useDeferredValue, useEffect, useMemo, useRef, useState,
} from 'react'
import { Link } from '@tanstack/react-router'
import { Search } from 'lucide-react'

import { BottomSheet } from '@shared/ui/BottomSheet'
import { Popover } from '@shared/ui/Popover'
import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { useIsDesktop } from '@shared/lib/useMediaQuery'
import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'
import { useReader } from '../ReaderContext'


// ── Helpers ────────────────────────────────────────────────────


function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}


/** Strip "Ch. N", "Ch.N", "Chapter N", "Chương N" prefixes from a
 *  chapter label so we don't render "Ch.5  Ch. 5" twice when the
 *  source already prefixes the title. Returns the leftover title, or
 *  null if nothing meaningful remains after stripping. */
function chapterSubtitle(number: string, label: string | null | undefined): string | null {
  if (!label) return null
  const trimmed = label.trim()
  if (!trimmed) return null
  const patterns = [
    new RegExp(`^(?:ch(?:apter|ương|uong)?)\\s*\\.?\\s*${escapeRegExp(number)}\\s*[:.\\-–—]?\\s*`, 'i'),
    /^(?:ch(?:apter|ương|uong)?)\s*\.?\s*\d+(?:\.\d+)?\s*[:.\-–—]?\s*/i,
  ]
  let rest = trimmed
  for (const re of patterns) {
    const next = rest.replace(re, '')
    if (next !== rest) { rest = next; break }
  }
  rest = rest.trim()
  if (!rest) return null
  if (rest === number) return null
  return rest
}


// ── Components ─────────────────────────────────────────────────


interface Props {
  open:    boolean
  onClose: () => void
  /** Trigger ref — popover anchors against this on desktop. */
  anchorRef: React.RefObject<HTMLButtonElement | null>
}


export function ChapterPicker({ open, onClose, anchorRef }: Props) {
  const isDesktop = useIsDesktop()

  if (isDesktop) {
    return (
      <Popover
        open={open}
        onClose={onClose}
        anchorRef={anchorRef}
        align="end"
        minWidth={320}
        maxWidth={360}
      >
        <Body onClose={onClose} />
      </Popover>
    )
  }

  return (
    <BottomSheet open={open} onClose={onClose} title="Danh sách chương">
      <Body onClose={onClose} />
    </BottomSheet>
  )
}


function Body({ onClose }: { onClose: () => void }) {
  const { merged } = useWorkChapters()
  const { workId, chapterRef } = useReader()

  const [q, setQ] = useState('')
  const term = useDeferredValue(q).trim().toLowerCase()

  const rows = useMemo(() => {
    if (!term) return merged
    return merged.filter(c =>
      `${c.number} ${c.label}`.toLowerCase().includes(term),
    )
  }, [merged, term])

  // Scroll current chapter into view on open
  const listRef = useRef<HTMLUListElement>(null)
  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-ref="${chapterRef}"]`)
    el?.scrollIntoView({ block: 'center' })
  }, [chapterRef])

  return (
    <div className="flex flex-col max-h-[60dvh] sm:max-h-[70vh]">
      <div className="px-3 py-2 sticky top-0 bg-surface z-10 border-b border-border-soft">
        <div className="relative">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none" />
          <input
            type="search"
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Tìm chương…"
            className={cn(inputCls, 'pl-8')}
            autoFocus
          />
        </div>
      </div>

      <ul ref={listRef} className="flex-1 overflow-y-auto p-2">
        {rows.length === 0 ? (
          <li className="px-3 py-8 text-center text-sm text-text-subtle">
            Không khớp tìm kiếm
          </li>
        ) : rows.map(c => {
          const active = c.numberNorm === chapterRef
          const subtitle = chapterSubtitle(c.number, c.label)
          return (
            <li key={c.numberNorm} data-ref={c.numberNorm}>
              <Link
                to="/r/$workId/$numberNorm"
                params={{ workId, numberNorm: c.numberNorm }}
                onClick={onClose}
                className={cn(
                  'relative flex items-center gap-3 h-9 px-3 rounded-sm text-sm',
                  'transition-colors duration-150 cursor-pointer',
                  active
                    ? 'bg-accent-bg text-accent-text font-medium'
                    : 'text-text-muted hover:text-text hover:bg-hover',
                )}
              >
                {active && (
                  <span
                    aria-hidden
                    className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-full bg-accent"
                  />
                )}
                <span className="tabular-nums font-medium shrink-0">
                  Ch.{c.number || c.numberNorm}
                </span>
                {subtitle && (
                  <span className={cn(
                    'truncate text-xs',
                    active ? 'text-accent-text' : 'text-text-subtle',
                  )}>
                    {subtitle}
                  </span>
                )}
              </Link>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
