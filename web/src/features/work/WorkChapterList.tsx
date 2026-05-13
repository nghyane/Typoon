// WorkChapterList — flat list of every readable version across every
// chapter in this Work.
//
// One row per (chapter, version). No grouping, no headers, no
// expand/collapse. Each row is a single tap/click action:
//
//   • [VI] @userA …       →   opens the translated reader
//   • [VI] @scanlator …   →   opens the raw reader
//   • [EN] @scanlator … ✨ →   spawns a translation (target_lang
//                              from the viewer's library entry)
//
// Toolbar:
//   • search input (deferred for typing smoothness)
//   • horizontal-scroll language chip rail (default: target_lang
//     only; tap "Tất cả" to surface every lang)
//   • sort tabs (Chương | Mới)
//
// Long lists (1k+ rows on Bleach/OP) are virtualized via
// `@tanstack/react-virtual` so scroll stays at 60fps regardless of
// total chapter count.

import {
  useDeferredValue, useEffect, useMemo, useRef, useState,
} from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  ArrowDown, ArrowUp, ChevronDown,
  PauseCircle, Search,
} from 'lucide-react'

import { cn } from '@shared/lib/cn'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { languageName } from '@shared/lib/lang'
import {
  preferredReadable, type HubChapter, type HubVersion,
} from '@features/title/mergeChapters'
import { useSpawnChapter, type SpawnProgress } from '@features/title/useSpawnChapter'

import { VersionLine, type VersionAction } from './VersionLine'


export interface WorkChapterListProps {
  chapters:        HubChapter[]
  targetLang:      string | null
  loading:         boolean
  spawnState:      SpawnProgress | null
  spawningKey:     string | null
  onSpawn:         (chapter: HubChapter, raw: HubVersion) => void
  /** Re-kick a translation that ended in `error`. Distinct from
   *  `onSpawn` — there's no raw to re-upload, the server already has
   *  the chapter bytes and we just POST `/translate/{id}/redo`. */
  onRetryTranslation: (translationId: number) => void
  onOpenVersion:   (chapter: HubChapter, v: HubVersion) => void
}


/** Sort axis for the chapter list — always by chapter number;
 *  only the direction toggles. */
type SortBy = 'newest' | 'oldest'


type Row = {
  chapter:       HubChapter
  /** The version whose identity (creator, date, kind chip) the row
   *  surfaces. For a done/in-flight/failed target translation this is
   *  the translation; for a raw read this is the raw. */
  version:       HubVersion
  /** When the primary version is a non-done translation (running /
   *  pending / error / blocked) we still want the user to be able to
   *  open SOMETHING. If a raw on the same chapter is readable, this
   *  carries it so the row click falls through to read-raw while the
   *  state chip shows progress/retry/blocked. */
  rawFallback:   HubVersion | null
  readyInTarget: boolean
  score:         number
}


export function WorkChapterList({
  chapters, targetLang, loading,
  spawnState, spawningKey,
  onSpawn, onRetryTranslation, onOpenVersion,
}: WorkChapterListProps) {
  const tgt = normalizeBcp(targetLang)

  // Counts per BCP-47 lang for the filter chip rail.
  const langCounts = useMemo(() => {
    const m = new Map<string, number>()
    for (const c of chapters) {
      for (const v of c.versions) m.set(v.lang, (m.get(v.lang) ?? 0) + 1)
    }
    return [...m.entries()].sort((a, b) => {
      if (a[0] === tgt) return -1
      if (b[0] === tgt) return  1
      return a[0].localeCompare(b[0])
    })
  }, [chapters, tgt])

  // Active lang filter — single-select. `null` = "Tất cả" (every
  // lang). Default to the viewer's target_lang so they land on the
  // version they actually read.
  const [activeLang, setActiveLang] = useState<string | null>(() => (
    tgt || null
  ))
  const [q,      setQ]      = useState('')
  const [sortBy, setSortBy] = useState<SortBy>('newest')

  // Defer the search term so each keystroke doesn't block the row
  // re-filter on huge chapter lists.
  const deferredQ = useDeferredValue(q)

  // Pre-compute search haystack per chapter once. Avoids rebuilding
  // the lowercase string on every keystroke.
  const haystacks = useMemo(() => {
    const m = new Map<HubChapter, string>()
    for (const c of chapters) {
      m.set(c, `${c.number} ${c.label ?? ''}`.toLowerCase())
    }
    return m
  }, [chapters])

  // Build rows: one row per chapter when the filter is target/all
  // (the chapter's translation state, if any, owns the row); the
  // legacy multi-row-per-chapter shape only kicks in when the user
  // filters by a non-target lang — there they're explicitly browsing
  // raws and want to see each scanlator option.
  //
  // No "in progress" bucket: a translation that's running / pending
  // / error / blocked stays at its natural chapter position and the
  // row chip describes the state. Raw fallback is attached so the
  // user can read the upstream chapter while waiting (or after an
  // error) without losing the retry affordance.
  const mainRows = useMemo(() => {
    const term = deferredQ.trim().toLowerCase()
    const targetMode = activeLang === null || activeLang === tgt
    const rows: Row[] = []
    for (const c of chapters) {
      if (term && !haystacks.get(c)!.includes(term)) continue
      const readyInTarget = preferredReadable(c, targetLang) !== null

      if (targetMode) {
        const row = pickTargetRow(c, tgt, readyInTarget)
        if (row) rows.push(row)
        continue
      }

      // Non-target filter — user is browsing raws in a specific lang.
      // Surface every raw of that lang as its own row so they can
      // pick scanlator/source.
      for (const v of c.versions) {
        if (v.lang !== activeLang) continue
        if (v.kind !== 'raw') continue
        rows.push({
          chapter: c, version: v, rawFallback: null, readyInTarget,
          score: versionScore(v, tgt),
        })
      }
    }
    const dir = sortBy === 'newest' ? 1 : -1
    rows.sort((a, b) => {
      if (a.chapter.sortKey !== b.chapter.sortKey) {
        return (b.chapter.sortKey - a.chapter.sortKey) * dir
      }
      return a.score - b.score
    })
    return rows
  }, [chapters, haystacks, deferredQ, activeLang, targetLang, tgt, sortBy])

  // System-health banner: surface when ANY chapter in this work has
  // a translation in `blocked` state. Inferred from rows (no extra
  // endpoint) — if the workers paused a stage, every in-flight draft
  // on that stage flips to `blocked` and the worker stamps the same
  // reason on each, so we can derive both "is there a problem" and
  // "what is it" from the existing payload.
  const blockedSummary = useMemo(() => {
    let count  = 0
    let reason: string | null = null
    for (const c of chapters) {
      for (const v of c.versions) {
        if (v.kind === 'translation' && v.state === 'blocked') {
          count += 1
          if (!reason && v.errorMessage) reason = v.errorMessage
        }
      }
    }
    return count > 0 ? { count, reason } : null
  }, [chapters])

  return (
    <section className="px-4 sm:px-6">
      <Toolbar
        q={q}
        setQ={setQ}
        sortBy={sortBy}
        setSortBy={setSortBy}
        langCounts={langCounts}
        activeLang={activeLang}
        setActiveLang={setActiveLang}
        tgt={tgt}
        count={mainRows.length}
      />

      {blockedSummary && (
        <HealthBanner
          count={blockedSummary.count}
          reason={blockedSummary.reason}
        />
      )}

      {loading && chapters.length === 0 ? (
        <div className="py-16 flex justify-center">
          <Spinner size={20} />
        </div>
      ) : mainRows.length === 0 ? (
        <EmptyView
          totalChapters={chapters.length}
          q={deferredQ}
          hasActiveFilter={activeLang !== null}
        />
      ) : (
        <VirtualList
          rows={mainRows}
          tgt={tgt}
          spawnState={spawnState}
          spawningKey={spawningKey}
          onSpawn={onSpawn}
          onRetryTranslation={onRetryTranslation}
          onOpenVersion={onOpenVersion}
        />
      )}
    </section>
  )
}


// ── Toolbar ────────────────────────────────────────────────────


function Toolbar({
  q, setQ, sortBy, setSortBy,
  langCounts, activeLang, setActiveLang,
  tgt, count,
}: {
  q:             string
  setQ:          (v: string) => void
  sortBy:        SortBy
  setSortBy:     (v: SortBy) => void
  langCounts:    Array<[string, number]>
  activeLang:    string | null
  setActiveLang: (v: string | null) => void
  tgt:           string
  count:         number
}) {
  const totalAll = langCounts.reduce((s, [, n]) => s + n, 0)
  return (
    <div
      className={cn(
        'sticky top-0 z-10 -mx-4 sm:-mx-6 px-4 sm:px-6',
        'pt-3 pb-2 bg-bg/95 backdrop-blur-xs',
        'flex items-center gap-2 flex-wrap sm:flex-nowrap',
      )}
    >
      {/* Search — primary affordance, takes the leftmost slot. Capped
          on desktop so it doesn't sprawl. */}
      <div className="relative flex-1 min-w-0 sm:max-w-xs order-1">
        <Search
          size={14}
          className="absolute left-2 top-1/2 -translate-y-1/2 text-text-subtle"
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

      {/* Lang picker — single-select dropdown. `null` = all langs. */}
      <LangPicker
        value={activeLang}
        onChange={setActiveLang}
        options={langCounts}
        tgt={tgt}
        totalAll={totalAll}
      />

      {/* Meta cluster — count + sort, pushed to the far right on
          desktop, wraps to second line on very narrow phones. */}
      <div className="ml-auto inline-flex items-center gap-2 text-xs text-text-subtle order-3">
        <span className="tabular shrink-0">{count} chương</span>
        <span className="text-border-soft">·</span>
        <SortToggle sortBy={sortBy} setSortBy={setSortBy} />
      </div>
    </div>
  )
}


/** Single-select language dropdown. `null` = every lang. Target lang
 *  is pinned to the top of the menu and rendered with accent so the
 *  user can spot their default in one glance. */
function LangPicker({
  value, onChange, options, tgt, totalAll,
}: {
  value:    string | null
  onChange: (v: string | null) => void
  options:  Array<[string, number]>
  tgt:      string
  totalAll: number
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: globalThis.MouseEvent) => {
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

  const triggerLabel = value === null ? 'Tất cả' : value.toUpperCase()
  const triggerCount = value === null
    ? totalAll
    : (options.find(([l]) => l === value)?.[1] ?? 0)
  const isTarget = value !== null && value === tgt

  return (
    <div ref={ref} className="relative inline-flex shrink-0 order-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        className={cn(
          'h-8 px-2.5 rounded-sm text-xs cursor-pointer transition-colors',
          'inline-flex items-center gap-1.5',
          'border border-border-soft bg-surface-2 hover:bg-hover',
          // Active state shows via text color only — no accent
          // background, no thicker border.
          isTarget          ? 'text-accent'
          : value !== null  ? 'text-text'
          :                   'text-text-muted hover:text-text',
        )}
      >
        <span className="tabular font-medium">{triggerLabel}</span>
        <span className="text-text-subtle tabular">{triggerCount}</span>
        <ChevronDown size={12} className="text-text-subtle" />
      </button>

      {open && (
        <div
          role="menu"
          className={cn(
            'absolute top-full left-0 mt-1 z-30 min-w-[220px]',
            'bg-surface rounded-md shadow-[0_8px_24px_rgb(0,0,0,0.35)]',
            'border border-border-soft py-1 max-h-[60vh] overflow-y-auto',
          )}
        >
          <LangOption
            active={value === null}
            label="Tất cả"
            count={totalAll}
            onSelect={() => { onChange(null); setOpen(false) }}
          />
          <div className="my-1 border-t border-border-soft/60" />
          {options.map(([lang, n]) => (
            <LangOption
              key={lang}
              active={value === lang}
              accent={lang === tgt}
              label={lang.toUpperCase()}
              sublabel={languageName(lang)}
              badge={lang === tgt ? 'Đích' : undefined}
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
        // Active state shows via text color only (matches the old
        // chip rail behavior — no background highlight).
        accent ? 'text-accent'
          : active ? 'text-text font-medium'
          : 'text-text-muted hover:text-text',
      )}
    >
      <span className="tabular font-medium shrink-0 whitespace-nowrap">
        {label}
      </span>
      {sublabel && (
        <span className="truncate text-text-subtle text-xs flex-1 min-w-0">
          {sublabel}
        </span>
      )}
      {badge && (
        <span className="text-[10px] px-1 h-4 inline-flex items-center rounded-xs text-accent shrink-0">
          {badge}
        </span>
      )}
      <span className="tabular text-xs text-text-subtle shrink-0 ml-auto">
        {count}
      </span>
    </button>
  )
}


// ── Virtual list ───────────────────────────────────────────────


function VirtualList({
  rows, tgt, spawnState, spawningKey, onSpawn, onRetryTranslation, onOpenVersion,
}: {
  rows:              Row[]
  tgt:               string
  spawnState:        SpawnProgress | null
  spawningKey:       string | null
  onSpawn:           (c: HubChapter, v: HubVersion) => void
  onRetryTranslation:(translationId: number) => void
  onOpenVersion:     (c: HubChapter, v: HubVersion) => void
}) {
  // AppLayout uses `<main className="flex-1 overflow-auto">` as the
  // page scroll container — NOT the window. Find it on mount via
  // ancestor walk so the virtualizer can observe the right element.
  const parentRef = useRef<HTMLDivElement>(null)
  const [scrollEl, setScrollEl] = useState<HTMLElement | null>(null)
  // Offset of the list relative to the scroll container's top. The
  // hero, source rail, description, toolbar, and (optionally) the
  // in-progress section all live ABOVE the list, so without this
  // virtualizer-tracked offset the first visible row would draw at
  // y=0 of the scroll container — i.e. behind the hero — and the
  // user would scroll into a big empty band before items show up.
  const [scrollMargin, setScrollMargin] = useState(0)

  useEffect(() => {
    let el: HTMLElement | null = parentRef.current
    while (el && el !== document.body) {
      const overflowY = getComputedStyle(el).overflowY
      if (overflowY === 'auto' || overflowY === 'scroll') {
        setScrollEl(el)
        break
      }
      el = el.parentElement
    }
    if (!el || el === document.body) {
      // Fall back to the document scrolling element (bare-chrome
      // routes where window itself scrolls).
      setScrollEl((document.scrollingElement as HTMLElement) ?? null)
    }
  }, [])

  // Measure how far down the scroll container the list sits. Tracked
  // via ResizeObserver because the hero (cover, description toggle)
  // changes height after image load and on user interaction.
  useEffect(() => {
    if (!scrollEl || !parentRef.current) return
    const measure = () => {
      if (!parentRef.current || !scrollEl) return
      const listTop   = parentRef.current.getBoundingClientRect().top
      const scrollTop = scrollEl.getBoundingClientRect().top
      setScrollMargin(
        listTop - scrollTop + (scrollEl.scrollTop ?? 0),
      )
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(scrollEl)
    if (parentRef.current.parentElement) {
      ro.observe(parentRef.current.parentElement)
    }
    return () => ro.disconnect()
  }, [scrollEl])

  // Track viewport size for estimateSize. Mobile rows are taller
  // (2-line layout, ~56px). Desktop rows are single-line (~44px).
  const [isWide, setIsWide] = useState(() => (
    typeof window !== 'undefined' && window.matchMedia('(min-width: 640px)').matches
  ))
  useEffect(() => {
    if (typeof window === 'undefined') return
    const mq = window.matchMedia('(min-width: 640px)')
    const onChange = () => setIsWide(mq.matches)
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  const virt = useVirtualizer({
    count:            rows.length,
    getScrollElement: () => scrollEl,
    estimateSize:     () => (isWide ? 44 : 56),
    overscan:         8,
    getItemKey:       (i) => rows[i]!.version.key,
    scrollMargin,
  })

  return (
    <div
      ref={parentRef}
      className="relative w-full"
      style={{ height: virt.getTotalSize() || undefined }}
    >
      {virt.getVirtualItems().map((vi) => {
        const row = rows[vi.index]!
        const { chapter, version, readyInTarget, rawFallback } = row
        const isSpawning = spawningKey === version.key
        const action = resolveAction(
          version, tgt, readyInTarget, isSpawning, spawnState, rawFallback,
        )
        return (
          <div
            key={vi.key}
            data-index={vi.index}
            ref={virt.measureElement}
            className="absolute left-0 right-0 border-b border-border-soft/60"
            style={{
              // `vi.start` is in scroll-container coordinates; the
              // list itself starts at `scrollMargin` from that
              // origin, so subtract to get local Y.
              transform: `translateY(${vi.start - virt.options.scrollMargin}px)`,
            }}
          >
            <VersionLine
              chapterNumber={chapter.number}
              version={version}
              action={action}
              progressLabel={
                isSpawning ? spawnLabel(spawnState) : undefined
              }
              errorMessage={
                // Prefer the live spawn-pipeline error (only meaningful
                // for the row currently being spawned by THIS client);
                // otherwise fall back to whatever the worker stamped on
                // the draft. Worker-stamped messages survive page
                // refresh and apply to every viewer, so they're the
                // authoritative source for `state ∈ {error,blocked}`.
                isSpawning && spawnState?.phase === 'error'
                  ? spawnState.error
                  : version.errorMessage
              }
              onClick={() => {
                if (action.kind === 'disabled') return
                // In-flight / failed / blocked translation rows: row
                // click opens the raw fallback if any (so the user can
                // read while waiting / after error). Retry / progress
                // / blocked indicators live on the chip.
                if (action.kind === 'spawn-pending'
                    || action.kind === 'spawn-progress'
                    || action.kind === 'spawn-blocked'
                    || action.kind === 'spawn-error') {
                  if (rawFallback) onOpenVersion(chapter, rawFallback)
                  return
                }
                if (action.kind === 'spawn-translate') {
                  onSpawn(chapter, version)
                } else if (action.kind === 'read-translation'
                        || action.kind === 'read-raw'
                        || action.kind === 'read-raw-with-spawn') {
                  onOpenVersion(chapter, version)
                }
              }}
              onSpawn={
                action.kind === 'read-raw-with-spawn'
                  && action.spawnState !== 'progress'
                  ? () => onSpawn(chapter, version)
                  : undefined
              }
              onRetry={
                action.kind === 'spawn-error'
                  && version.kind === 'translation'
                  && version.translationId != null
                  ? () => onRetryTranslation(version.translationId!)
                  : undefined
              }
            />
          </div>
        )
      })}
    </div>
  )
}


// ── Action resolution ───────────────────────────────────────────


/** Decide what clicking a row does given the (primary) version +
 *  chapter context. When the primary is an in-flight / failed
 *  translation, `rawFallback` is the raw the row falls back to so the
 *  user can read while waiting; the state chip stays the actionable
 *  affordance for retry / progress / blocked. */
function resolveAction(
  v:             HubVersion,
  targetLang:    string,
  readyInTarget: boolean,
  isSpawning:    boolean,
  progress:      SpawnProgress | null,
  rawFallback:   HubVersion | null,
): VersionAction {
  if (v.kind === 'translation') {
    const fb = rawFallback !== null
    if (v.state === 'pending') return { kind: 'spawn-pending',  rawFallback: fb }
    if (v.state === 'running') return { kind: 'spawn-progress', rawFallback: fb }
    if (v.state === 'blocked') return { kind: 'spawn-blocked',  rawFallback: fb }
    if (v.state === 'error')   return { kind: 'spawn-error',    rawFallback: fb }
    return { kind: 'read-translation' }
  }
  // raw
  if (!v.upstreamUrl || !v.sourceId) {
    return { kind: 'disabled', reason: 'Plugin nguồn chưa cài.' }
  }
  if (v.lang === targetLang) return { kind: 'read-raw' }
  if (!targetLang)           return { kind: 'read-raw' }
  if (readyInTarget)         return { kind: 'read-raw' }
  // Non-target raw with no target translation yet → row has BOTH a
  // read-raw action (the row's click) AND a separate spawn affordance
  // (the chip). The chip's onClick stops propagation so the user can
  // pick "read upstream language verbatim" vs "translate" without
  // ambiguity. See `onChipClick` plumbing in `VersionLine`.
  if (isSpawning) {
    return progress?.phase === 'error'
      ? { kind: 'read-raw-with-spawn', spawnState: 'error' }
      : { kind: 'read-raw-with-spawn', spawnState: 'progress' }
  }
  return { kind: 'read-raw-with-spawn', spawnState: 'idle' }
}


/** Pick the best translation among candidates for a given target lang.
 *  Priority: done > running > pending > error > blocked. Ties broken
 *  by `date` desc so the most recent attempt wins.
 *
 *  Why this order: the user-visible answer to "is this chapter ready
 *  in <lang>?" is the strongest yes (done), then "soon" (in-flight),
 *  then "needs me" (error), then "needs admin" (blocked). Putting
 *  blocked last keeps a single stuck row from masking a viable retry. */
function pickBestTranslation(
  versions: HubVersion[],
  lang:     string,
): HubVersion | null {
  const stateRank = (v: HubVersion): number => {
    if (v.state === 'done')    return 0
    if (v.state === 'running') return 1
    if (v.state === 'pending') return 2
    if (v.state === 'error')   return 3
    if (v.state === 'blocked') return 4
    return 5
  }
  let best: HubVersion | null = null
  for (const v of versions) {
    if (v.kind !== 'translation' || v.lang !== lang) continue
    if (!best) { best = v; continue }
    const a = stateRank(v)
    const b = stateRank(best)
    if (a < b) { best = v; continue }
    if (a > b) continue
    // Same state → newer wins.
    if ((v.date ?? '') > (best.date ?? '')) best = v
  }
  return best
}


/** Pick the best raw to read on a chapter — prefers target lang, then
 *  any installed-source raw with an upstream URL. Returns null when
 *  nothing is readable (raw missing / source uninstalled). */
function pickBestRaw(
  versions: HubVersion[],
  tgt:      string,
): HubVersion | null {
  let best: HubVersion | null = null
  let bestScore = Infinity
  for (const v of versions) {
    if (v.kind !== 'raw') continue
    if (!v.upstreamUrl || !v.sourceId) continue
    const s = versionScore(v, tgt)
    if (s < bestScore) { best = v; bestScore = s }
  }
  return best
}


/** Build a single row for a chapter in target/all filter mode.
 *
 *  Policy: one row per chapter. The row's PRIMARY version owns the
 *  identity (creator, date, kind chip). When a target translation
 *  exists in any state, it's the primary — including running / error
 *  / blocked, so the user sees "their" translation row at the natural
 *  chapter position with a state chip. A raw fallback is attached so
 *  the row click can still open something readable while the worker
 *  finishes (or after an error).
 *
 *  Returns null only when the chapter has nothing at all to surface
 *  (no target translation, no installed-source raw, and no
 *  spawn-eligible raw to translate from). */
function pickTargetRow(
  c:             HubChapter,
  tgt:           string,
  readyInTarget: boolean,
): Row | null {
  // 1. Best target-lang translation (any state).
  const trans = tgt ? pickBestTranslation(c.versions, tgt) : null
  if (trans) {
    const fallback = trans.state === 'done' ? null : pickBestRaw(c.versions, tgt)
    return {
      chapter: c, version: trans, rawFallback: fallback, readyInTarget,
      score:   versionScore(trans, tgt),
    }
  }

  // 2. No translation — fall back to raw. Prefer target-lang raw so
  //    "VI raw" beats "EN raw" on the row.
  if (tgt) {
    const targetRaw = c.versions.find(
      (v) => v.kind === 'raw' && v.lang === tgt
          && v.upstreamUrl && v.sourceId,
    ) ?? null
    if (targetRaw) {
      return {
        chapter: c, version: targetRaw, rawFallback: null, readyInTarget,
        score:   versionScore(targetRaw, tgt),
      }
    }
  }

  // 3. Other-lang raw — surfaced so the user has a "Dịch" affordance.
  const otherRaw = pickBestRaw(c.versions, tgt)
  if (otherRaw) {
    return {
      chapter: c, version: otherRaw, rawFallback: null, readyInTarget,
      score:   versionScore(otherRaw, tgt),
    }
  }
  return null
}


function spawnLabel(p: SpawnProgress | null): string {
  if (!p) return 'Đang dịch…'
  switch (p.phase) {
    case 'fetching':    return 'Lấy trang…'
    case 'downloading': return `${p.current}/${p.total}`
    case 'packing':     return 'Đóng gói…'
    case 'uploading':   return `Tải lên ${p.pct}%`
    case 'spawning':    return 'Khởi tạo…'
    default:            return 'Đang dịch…'
  }
}


/** Smaller score → row sorts earlier within the same chapter.
 *  Target-lang reads first; spawn-eligible raws last. */
function versionScore(v: HubVersion, targetLang: string): number {
  const isTarget = v.lang === targetLang
  if (v.kind === 'translation') {
    if (v.state === 'done')                            return isTarget ? 0 : 10
    if (v.state === 'pending' || v.state === 'running') return isTarget ? 1 : 11
    return isTarget ? 2 : 12  // error
  }
  return isTarget ? 1 : 20
}


function normalizeBcp(code: string | null): string {
  if (!code) return ''
  return code.toLowerCase().split(/[-_]/)[0]!
}


/** Single toggle button that flips chapter sort direction. Label
 *  reflects the CURRENT order so the user reads "what they see"
 *  rather than "what will happen on click". */
function SortToggle({
  sortBy, setSortBy,
}: {
  sortBy:    SortBy
  setSortBy: (v: SortBy) => void
}) {
  const newest = sortBy === 'newest'
  return (
    <button
      type="button"
      onClick={() => setSortBy(newest ? 'oldest' : 'newest')}
      title={newest ? 'Đang xếp: mới nhất trước' : 'Đang xếp: cũ nhất trước'}
      className={cn(
        'shrink-0 h-7 px-1.5 rounded-sm text-xs cursor-pointer transition-colors',
        'inline-flex items-center gap-1 text-text-muted hover:text-text hover:bg-hover',
      )}
    >
      {newest ? <ArrowDown size={12} /> : <ArrowUp size={12} />}
      <span>{newest ? 'Mới nhất' : 'Cũ nhất'}</span>
    </button>
  )
}


/** System-health banner shown above the chapter list when at least
 *  one translation is in `blocked` state. Tells the reader they
 *  haven't done anything wrong — the pipeline is paused for admin
 *  attention — so they don't spam-click "Dịch" trying to recover.
 *  Inferred from the chapters payload; no separate health endpoint.
 */
function HealthBanner({
  count, reason,
}: {
  count:  number
  reason: string | null
}) {
  return (
    <div
      className={cn(
        'mb-3 rounded-md border border-amber-500/30 bg-amber-500/10',
        'px-3 py-2 flex items-start gap-2',
      )}
      role="status"
    >
      <PauseCircle
        size={16}
        className="shrink-0 mt-0.5 text-amber-400"
      />
      <div className="min-w-0 text-xs sm:text-sm text-amber-200/90">
        <span className="font-medium text-amber-200">
          Hệ thống dịch tạm ngưng
        </span>
        <span className="text-amber-200/70">
          {' · '}{count} chương đang chờ quản trị xử lý.
        </span>
        {reason && (
          <div className="mt-0.5 text-amber-200/60 truncate" title={reason}>
            {humanizeBlockedReason(reason)}
          </div>
        )}
      </div>
    </div>
  )
}


/** Map the worker's raw error into the same short phrase
 *  `VersionLine` uses, so the banner reads consistently with the
 *  row chips below it. Kept here (not in VersionLine) so the banner
 *  works without the row context. */
function humanizeBlockedReason(raw: string): string {
  const low = raw.toLowerCase()
  if (low.includes('model_not_found') || low.includes('model not found')) {
    return 'Model dịch chưa khả dụng — quản trị đang cập nhật cấu hình.'
  }
  if (low.includes('no available credential')
      || low.includes('credential has been invalidated')
      || low.includes('token_invalidated')) {
    return 'Khoá API hết hạn — quản trị đang thay khoá mới.'
  }
  if (low.includes('insufficient_quota')
      || low.includes('billing_hard_limit')
      || low.includes('account is suspended')) {
    return 'Tài khoản dịch hết hạn mức — quản trị đang nạp lại.'
  }
  if (low.includes('region not supported')) {
    return 'Khu vực không hỗ trợ — quản trị đang đổi tuyến.'
  }
  return 'Quản trị viên đã được thông báo, các chương sẽ tiếp tục khi xử lý xong.'
}


// ── Empty view ─────────────────────────────────────────────────


function EmptyView({
  totalChapters, q, hasActiveFilter,
}: {
  totalChapters:   number
  q:               string
  hasActiveFilter: boolean
}) {
  return (
    <EmptyState
      icon={Search}
      title={
        totalChapters === 0
          ? 'Chưa có chương nào'
          : hasActiveFilter
          ? 'Không có chương phù hợp ngôn ngữ đã chọn'
          : 'Không tìm thấy chương phù hợp'
      }
      hint={
        totalChapters === 0
          ? 'Nguồn chưa trả về chương nào. Cài plugin khác hoặc đợi cập nhật.'
          : q
          ? 'Thử từ khoá khác.'
          : hasActiveFilter
          ? 'Bấm "Tất cả" để xem mọi ngôn ngữ.'
          : undefined
      }
    />
  )
}


// useChapterSpawn — wrapper around `useSpawnChapter` that exposes
// "which version is currently spawning" so only that row reflects
// progress. Key is HubVersion.key for the raw being translated.
export function useChapterSpawn(targetLang: string | null) {
  const lang = targetLang ?? ''
  const { progress, spawn, reset } = useSpawnChapter(lang)
  const [spawningKey, setSpawningKey] = useState<string | null>(null)

  return {
    progress,
    spawningKey,
    spawn: (version: HubVersion, label: string | null) => {
      setSpawningKey(version.key)
      spawn(version, label)
    },
    reset: () => {
      reset()
      setSpawningKey(null)
    },
  }
}
