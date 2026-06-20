// ChapterToolbar — sticky controls above the virtualized chapter list.
//
// Slots, left → right:
//
//   [ Search input             ] [ Lang ▾ ] [ Mới nhất ↓ ]      N chương
//
// All four atoms are visible by default — no kebab. Mobile users
// shouldn't have to learn a hidden menu to flip sort order. Sort is
// a tap-toggle (only two options) with an arrow that mirrors the
// direction, so the control simultaneously shows current state and
// the action when tapped.
//
// Reader actions live inside the reader; this toolbar stays focused on
// filtering and chapter navigation.
//
// `langCounts` is the per-lang chapter count across every source.
// Languages with 0 chapters never appear so the picker stays terse.
// `null` filter = "Tự động" (every lang); the target lang is pinned
// to the top of the picker with accent so the viewer's default
// reading lang is one glance away.

import { useEffect, useRef, useState } from 'react'
import { ArrowDown, ArrowUp, ChevronDown, Search } from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { languageName } from '@shared/lib/lang'


export type SortBy = 'newest' | 'oldest'


interface Props {
  q:             string
  setQ:          (v: string) => void
  sortBy:        SortBy
  setSortBy:     (v: SortBy) => void
  langCounts:    Array<[string, number]>
  activeLang:    string | null
  setActiveLang: (v: string | null) => void
  /** Viewer's target language — pinned to the top of the picker
   *  with accent treatment. */
  targetLang:    string
  /** Visible-row count after search + lang filter. */
  count:         number
}

export function ChapterToolbar({
  q, setQ, sortBy, setSortBy,
  langCounts, activeLang, setActiveLang,
  targetLang, count,
}: Props) {
  const totalAll = langCounts.reduce((s, [, n]) => s + n, 0)

  return (
    <div
      className={cn(
        'sticky top-0 z-10 pt-3 pb-2 bg-bg/95 border-b border-border-soft/60',
        'flex items-center gap-2 flex-wrap sm:flex-nowrap',
      )}
    >
      <div className="relative flex-1 min-w-0 sm:max-w-xs order-1">
        <Search
          size={14}
          className="absolute left-2 top-1/2 -translate-y-1/2 text-text-subtle pointer-events-none"
        />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Tìm chương…"
          className={cn(
            'w-full h-8 pl-7 pr-2 rounded-sm text-sm',
            'bg-surface-2 text-text placeholder:text-text-subtle',
            'focus:outline-hidden focus:ring-1 focus:ring-accent/40',
          )}
        />
      </div>

      <LangPicker
        value={activeLang}
        onChange={setActiveLang}
        options={langCounts}
        targetLang={targetLang}
        totalAll={totalAll}
      />

      <div className="ml-auto flex items-center gap-2 shrink-0 order-4">
        <span className="text-xs text-text-subtle tabular-nums">{count} chương</span>
        <span className="text-text-subtle">·</span>
        <SortToggle sortBy={sortBy} setSortBy={setSortBy} />
      </div>
    </div>
  )
}


// ── SortToggle ───────────────────────────────────────
//
// Two-option toggle as a single chip. Arrow encodes current state
// (↓ = newest first, ↑ = oldest first) and a tap flips it.


function SortToggle({
  sortBy, setSortBy,
}: {
  sortBy:    SortBy
  setSortBy: (v: SortBy) => void
}) {
  const isNewest = sortBy === 'newest'
  return (
    <button
      type="button"
      onClick={() => setSortBy(isNewest ? 'oldest' : 'newest')}
      className="inline-flex items-center gap-1 text-xs text-text-subtle hover:text-text transition-colors cursor-pointer"
    >
      {isNewest ? 'Mới nhất' : 'Cũ nhất'}
      {isNewest
        ? <ArrowDown size={11} />
        : <ArrowUp   size={11} />}
    </button>
  )
}


// ── LangPicker ──────────────────────────────────────────────────


function LangPicker({
  value, onChange, options, targetLang, totalAll,
}: {
  value:      string | null
  onChange:   (v: string | null) => void
  options:    Array<[string, number]>
  targetLang: string
  totalAll:   number
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const triggerLabel = value === null ? 'Tự động' : value.toUpperCase()
  const triggerCount = value === null
    ? totalAll
    : (options.find(([l]) => l === value)?.[1] ?? 0)
  const isTarget = value !== null && value === targetLang

  return (
    <div ref={ref} className="relative inline-flex shrink-0 order-2">
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(
          'h-8 px-3 rounded-sm text-xs cursor-pointer transition-colors',
          'inline-flex items-center gap-2',
          'border border-border-soft bg-surface-2 hover:bg-hover',
          isTarget          ? 'text-accent'
          : value !== null  ? 'text-text'
          :                   'text-text-muted hover:text-text',
        )}
      >
        <span className="tabular-nums font-medium">{triggerLabel}</span>
        <span className="text-text-subtle tabular-nums">{triggerCount}</span>
        <ChevronDown size={12} className="text-text-subtle" />
      </button>

      {open && (
        <div
          role="menu"
          className={cn(
            'absolute top-full left-0 mt-1 z-30 min-w-[220px]',
            'bg-surface rounded-md border border-border-soft',
            'border border-border-soft py-1 max-h-[60vh] overflow-y-auto',
          )}
        >
          <LangOption
            active={value === null}
            label="Tự động"
            count={totalAll}
            onSelect={() => { onChange(null); setOpen(false) }}
          />
          <div className="my-1 border-t border-border-soft/60" />
          {options.map(([lang, n]) => (
            <LangOption
              key={lang}
              active={value === lang}
              accent={lang === targetLang}
              label={lang.toUpperCase()}
              sublabel={languageName(lang)}
              badge={lang === targetLang ? 'Đích' : undefined}
              count={n}
              onSelect={() => { onChange(lang); setOpen(false) }}
            />
          ))}
        </div>
      )}
    </div>
  )
}


function LangOption({
  active, accent, label, sublabel, count, badge, onSelect,
}: {
  active:    boolean
  accent?:   boolean
  label:     string
  sublabel?: string
  count:     number
  badge?:    string
  onSelect:  () => void
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onSelect}
      className={cn(
        'w-full flex items-center gap-2 px-3 py-1.5 text-sm text-left',
        'transition-colors cursor-pointer hover:bg-hover',
        accent     ? 'text-accent'
        : active   ? 'text-text font-medium'
        :            'text-text-muted hover:text-text',
      )}
    >
      <span className="tabular-nums font-medium shrink-0 whitespace-nowrap">{label}</span>
      {sublabel && (
        <span className="truncate text-text-subtle text-xs flex-1 min-w-0">{sublabel}</span>
      )}
      {badge && (
        <span className="text-xs px-1 h-4 inline-flex items-center rounded-xs text-accent shrink-0">
          {badge}
        </span>
      )}
      <span className="tabular-nums text-xs text-text-subtle shrink-0 ml-auto">{count}</span>
    </button>
  )
}
