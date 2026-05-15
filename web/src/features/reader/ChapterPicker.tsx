// ChapterPicker — drop-down list of all chapters in the work.
// Desktop: anchored popover from the chapter chip on the top bar.
// Mobile:  bottom sheet with search at the top.
//
// One row per chapter. Visual model:
//
//   ┌────────────────────────────────┐
//   │ 🔍 Tìm chương…                 │
//   ├────────────────────────────────┤
//   │ Ch.93              [VI]        │
//   │ Ch.92              […]         │
//   │ ▶ Ch.5 (đang đọc)  [VI]       │
//   │ Ch.4               [!]         │
//   │ Ch.3               [EN]        │   ← raw-only, can spawn
//   └────────────────────────────────┘
//
// State chip per row summarises the default version's readiness so
// the user can scan the list at a glance:
//
//   [VI]   — readable in target lang (raw scan or AI done)
//   […]    — AI mid-pipeline / pending
//   [!]    — AI error, retry from work hub
//   [EN]   — only non-target raw exists; tap to read, spawn from
//            source picker
//   [—]    — nothing readable; filler / no installed source
//
// The list shows the full spine (latest-first). Search filters by
// chapter number / label. Current chapter scrolls into view on open.

import {
  useDeferredValue, useEffect, useMemo, useRef, useState,
} from 'react'
import { Link } from '@tanstack/react-router'
import { Search } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { MenuShell } from '@shared/ui/MenuShell'
import type { HubChapter } from '@features/title/mergeChapters'

import { summarizeChapter } from './resolvers'
import { StateChip } from './atoms'
import { useReaderSettings, sourcePrefFor } from './store'


interface Props {
  open:        boolean
  onClose:     () => void
  anchorRef:   React.RefObject<HTMLElement | null>
  workId:      number
  chapters:    HubChapter[]
  currentNum:  string
  targetLang:  string | null
}


export function ChapterPicker({
  open, onClose, anchorRef, workId, chapters, currentNum, targetLang,
}: Props) {
  const pref = useReaderSettings((s) => sourcePrefFor(s, workId))

  const [q, setQ] = useState('')
  // Defer the search term so each keystroke doesn't block the
  // row-filter pass on huge spines (works with 1k chapters).
  const term = useDeferredValue(q).trim().toLowerCase()

  const rows = useMemo(() => {
    const out: Array<{
      chapter: HubChapter
      summary: ReturnType<typeof summarizeChapter>
    }> = []
    for (const c of chapters) {
      if (term) {
        const hay = `${c.number} ${c.label ?? ''}`.toLowerCase()
        if (!hay.includes(term)) continue
      }
      out.push({
        chapter: c,
        summary: summarizeChapter(c, targetLang, pref),
      })
    }
    return out
  }, [chapters, term, targetLang, pref])

  return (
    <MenuShell
      open={open}
      onClose={onClose}
      anchorRef={anchorRef}
      title="Chương"
      align="end"
      minWidth={300}
      maxWidth={360}
    >
      <div className="flex flex-col min-h-0 max-h-[60dvh] sm:max-h-[min(60dvh,32rem)]">
        <SearchInput value={q} onChange={setQ} />
        {rows.length === 0 ? (
          <p className="px-4 py-8 text-sm text-text-subtle text-center">
            Không tìm thấy chương khớp.
          </p>
        ) : (
          <ChapterList
            rows={rows}
            workId={workId}
            currentNum={currentNum}
            onPick={onClose}
            open={open}
          />
        )}
      </div>
    </MenuShell>
  )
}


// ── Search ────────────────────────────────────────────────────


function SearchInput({
  value, onChange,
}: {
  value:    string
  onChange: (v: string) => void
}) {
  return (
    <div className="shrink-0 px-3 pt-3 pb-2">
      <div
        className={cn(
          'flex items-center gap-2 px-3 h-9 rounded-md',
          'bg-surface-2 text-text',
          'focus-within:bg-hover transition-colors duration-150',
        )}
      >
        <Search size={14} className="shrink-0 text-text-subtle" />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Tìm chương…"
          className={cn(
            'flex-1 min-w-0 bg-transparent outline-none',
            'text-sm text-text placeholder:text-text-subtle',
          )}
          autoFocus
        />
      </div>
    </div>
  )
}


// ── List ──────────────────────────────────────────────────────


function ChapterList({
  rows, workId, currentNum, onPick, open,
}: {
  rows: Array<{
    chapter: HubChapter
    summary: ReturnType<typeof summarizeChapter>
  }>
  workId:     number
  currentNum: string
  onPick:     () => void
  open:       boolean
}) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const currentRef = useRef<HTMLAnchorElement>(null)

  // Scroll the current chapter into view when the picker opens.
  // Defer one frame so the panel has been measured & laid out.
  useEffect(() => {
    if (!open) return
    const t = requestAnimationFrame(() => {
      currentRef.current?.scrollIntoView({ block: 'center', behavior: 'auto' })
    })
    return () => cancelAnimationFrame(t)
  }, [open])

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto overscroll-contain"
    >
      <ul role="list" className="py-1">
        {rows.map(({ chapter, summary }) => {
          const isCurrent = chapter.number === currentNum
          return (
            <li key={chapter.number}>
              <Link
                ref={isCurrent ? currentRef : undefined}
                to="/r/$workId/$numberNorm"
                params={{ workId: String(workId), numberNorm: chapter.number }}
                onClick={onPick}
                className={cn(
                  'flex items-center gap-3 px-4 py-2.5',
                  'border-l-2 transition-colors duration-150',
                  isCurrent
                    // Active row: neutral elevation + accent indicator
                    // on the left edge. Avoid coloring the background
                    // with the accent (accent is reserved for CTAs);
                    // the surface lift + accent rail signal "you are
                    // here" without competing with action buttons.
                    ? 'bg-row-active border-l-accent'
                    : 'border-l-transparent hover:bg-hover',
                )}
              >
                <span
                  className={cn(
                    'shrink-0 w-12 tabular text-sm font-medium',
                    isCurrent ? 'text-accent' : 'text-text',
                  )}
                >
                  Ch.{chapter.number}
                </span>

                <div className="min-w-0 flex-1">
                  {chapter.label ? (
                    <p
                      className={cn(
                        'text-sm truncate',
                        isCurrent ? 'text-text' : 'text-text-muted',
                      )}
                    >
                      {chapter.label}
                    </p>
                  ) : (
                    <span aria-hidden />
                  )}
                </div>

                {/* Only flag abnormal state. A `done` row in the
                    target lang is the expected baseline \u2014 showing a
                    [VI] chip on every row would be noise. Surface
                    only running / error / raw-only / none so the
                    user's eye lands on the chapters that actually
                    need attention. */}
                {summary.state !== 'done' && (
                  <StateChip state={summary.state} label={summary.label} />
                )}
              </Link>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
