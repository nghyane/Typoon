// WorkChapterList — chapter-row view of every readable chapter in
// the Work.
//
// One chapter = one row = one primary button. The button is a state
// machine (read / translate / progress / error / blocked) keyed by
// chapter number, so the SAME row tracks every transition from raw
// → client-download → upload → server pending → done. No new rows
// appear next to the old one when a spawn lands.
//
// Toolbar: search + lang filter + sort. The filter is the "I want to
// see chapters readable in lang X" axis — `vi` shows chapters where
// the viewer's target_lang is reachable (translation done or native
// scanlation); other langs show chapters with a raw at that lang.
// Defaults to target_lang when at least one chapter is reachable
// there, otherwise "Tất cả".
//
// Long lists (1k+ rows on Bleach/OP) stay smooth via
// `@tanstack/react-virtual` measuring against the AppLayout scroll
// container.

import {
  useCallback, useDeferredValue, useEffect, useMemo, useRef, useState,
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
import { useSources } from '@features/browse/sources'

import {
  type HubChapter, type HubVersion,
} from '@features/title/mergeChapters'
import {
  deriveChapterRows, type ChapterRow as ChapterRowModel,
} from '@features/title/chapterRow'
import { useSpawnChapters, type SpawnProgress } from '@features/title/useSpawnChapter'

import { ChapterRow } from './ChapterRow'
import {
  useChapterListUiActions, useWorkListUi,
} from './chapterListUi'


export interface WorkChapterListProps {
  workId:             number
  chapters:           HubChapter[]
  targetLang:         string | null
  loading:            boolean
  /** Per-chapter spawn state lookup keyed by `chapter.number`. */
  getSpawnState:      (chapterNumber: string) => SpawnProgress | null
  onSpawn:            (chapter: HubChapter, raw: HubVersion) => void
  onAbort:            (chapter: HubChapter) => void
  /** Re-kick a server-side `error` translation. Distinct from
   *  `onSpawn` — there's no raw to re-upload, the server already has
   *  the chapter bytes and we POST `/translate/{id}/redo`. */
  onRetryTranslation: (translationId: number) => void
  onOpenVersion:      (chapter: HubChapter, v: HubVersion) => void
}


type SortBy = 'newest' | 'oldest'


export function WorkChapterList({
  workId, chapters, targetLang, loading,
  getSpawnState, onSpawn, onAbort, onRetryTranslation, onOpenVersion,
}: WorkChapterListProps) {
  const tgt = normalizeBcp(targetLang)
  const installedMap = useSources((s) => s.sources)
  const installedSourceIds = useMemo(
    () => new Set(Object.keys(installedMap)),
    [installedMap],
  )

  // Derive one ChapterRowModel per chapter. Pure fold — re-runs only
  // when chapter data or target lang changes.
  const allRows = useMemo<ChapterRowModel[]>(
    () => deriveChapterRows(chapters, tgt, { installedSourceIds }),
    [chapters, tgt, installedSourceIds],
  )

  // Counts per BCP-47 lang for the filter dropdown.
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

  // Persisted per-work UI state. `null` from the store = user hasn't
  // pinned anything yet → fall back to the default policy below.
  // Edits go through the store actions so a remount (reader → back)
  // reads the same pick the user just made.
  const persisted = useWorkListUi(workId)
  const { setActiveLang: storeSetActiveLang, setSortBy: storeSetSortBy }
    = useChapterListUiActions()

  // Default-lang policy: filter target_lang IF the work has anything
  // reachable there (translation done OR native scanlation), else
  // "Tất cả". Computed only when the user has no pinned pick — once
  // they pick, store wins.
  const defaultLang = useMemo<string | null>(() => {
    if (!tgt) return null
    if (chapters.length === 0) return null
    const hasTargetReachable = allRows.some(
      (r) => r.status.kind === 'read-translation'
          || r.status.kind === 'read-raw-target',
    )
    return hasTargetReachable ? tgt : null
  }, [tgt, chapters.length, allRows])

  const activeLang: string | null = persisted
    ? persisted.activeLang
    : defaultLang
  const sortBy: SortBy = persisted?.sortBy ?? 'newest'

  const setActiveLang = useCallback(
    (lang: string | null) => storeSetActiveLang(workId, lang),
    [workId, storeSetActiveLang],
  )
  const setSortBy = useCallback(
    (next: SortBy) => storeSetSortBy(workId, next),
    [workId, storeSetSortBy],
  )

  const [q, setQ] = useState('')
  const deferredQ = useDeferredValue(q)

  // Pre-compute search haystack per chapter.
  const haystacks = useMemo(() => {
    const m = new Map<HubChapter, string>()
    for (const c of chapters) {
      m.set(c, `${c.number} ${c.label ?? ''}`.toLowerCase())
    }
    return m
  }, [chapters])

  // Filter + sort the chapter rows. Per-lang filter semantics:
  //   • null    : every chapter.
  //   • tgt     : chapters where the row reads in target lang (done
  //               translation OR native scanlation).
  //   • other   : chapters with at least one raw at that lang.
  const rows = useMemo(() => {
    const term = deferredQ.trim().toLowerCase()
    const out: ChapterRowModel[] = []
    for (const r of allRows) {
      if (term && !haystacks.get(r.chapter)!.includes(term)) continue

      if (activeLang === null) {
        out.push(r)
        continue
      }
      if (activeLang === tgt) {
        // Show chapters readable at target lang, plus chapters that
        // are in-pipeline for target lang (translating, error, blocked,
        // or still being prepared from an upload). Hiding these would
        // make the user think nothing happened after they kicked a job.
        if (r.status.kind === 'read-translation'
         || r.status.kind === 'read-raw-target'
         || r.status.kind === 'translating-server'
         || r.status.kind === 'translation-error'
         || r.status.kind === 'translation-blocked'
         || r.status.kind === 'preparing') {
          out.push(r)
        }
        continue
      }
      // Non-target filter: chapter has at least one raw at that lang.
      if (r.chapter.versions.some(
        (v) => v.kind === 'raw' && v.lang === activeLang,
      )) {
        out.push(r)
      }
    }
    const dir = sortBy === 'newest' ? 1 : -1
    out.sort((a, b) => (b.chapter.sortKey - a.chapter.sortKey) * dir)
    return out
  }, [allRows, haystacks, deferredQ, activeLang, tgt, sortBy])

  // System-health banner: any chapter with a `blocked` translation.
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
        count={rows.length}
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
      ) : rows.length === 0 ? (
        <EmptyView
          totalChapters={chapters.length}
          q={deferredQ}
          hasActiveFilter={activeLang !== null}
        />
      ) : (
        <VirtualList
          rows={rows}
          getSpawnState={getSpawnState}
          onSpawn={onSpawn}
          onAbort={onAbort}
          onRetryTranslation={onRetryTranslation}
          onOpenVersion={onOpenVersion}
        />
      )}
    </section>
  )
}


// ── Virtual list ───────────────────────────────────────────────


function VirtualList({
  rows, getSpawnState, onSpawn, onAbort, onRetryTranslation, onOpenVersion,
}: {
  rows:               ChapterRowModel[]
  getSpawnState:      (chapterNumber: string) => SpawnProgress | null
  onSpawn:            (c: HubChapter, v: HubVersion) => void
  onAbort:            (c: HubChapter) => void
  onRetryTranslation: (translationId: number) => void
  onOpenVersion:      (c: HubChapter, v: HubVersion) => void
}) {
  // AppLayout uses `<main className="flex-1 overflow-auto">` as the
  // page scroll container — NOT the window. Find it via ancestor
  // walk so the virtualizer can observe the right element.
  const parentRef = useRef<HTMLDivElement>(null)
  const [scrollEl, setScrollEl] = useState<HTMLElement | null>(null)
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
      setScrollEl((document.scrollingElement as HTMLElement) ?? null)
    }
  }, [])

  useEffect(() => {
    if (!scrollEl || !parentRef.current) return
    const measure = () => {
      if (!parentRef.current || !scrollEl) return
      const listTop   = parentRef.current.getBoundingClientRect().top
      const scrollTop = scrollEl.getBoundingClientRect().top
      setScrollMargin(listTop - scrollTop + (scrollEl.scrollTop ?? 0))
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(scrollEl)
    if (parentRef.current.parentElement) {
      ro.observe(parentRef.current.parentElement)
    }
    return () => ro.disconnect()
  }, [scrollEl])

  const virt = useVirtualizer({
    count:            rows.length,
    getScrollElement: () => scrollEl,
    estimateSize:     () => 56,
    overscan:         8,
    getItemKey:       (i) => rows[i]!.chapter.number,
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
        return (
          <div
            key={vi.key}
            data-index={vi.index}
            ref={virt.measureElement}
            className="absolute left-0 right-0 border-b border-border-soft/60"
            style={{
              transform: `translateY(${vi.start - virt.options.scrollMargin}px)`,
            }}
          >
            <ChapterRow
              row={row}
              spawn={getSpawnState(row.chapter.number)}
              onRead={(v) => onOpenVersion(row.chapter, v)}
              onSpawn={(raw) => onSpawn(row.chapter, raw)}
              onAbort={() => onAbort(row.chapter)}
              onRetryServer={onRetryTranslation}
            />
          </div>
        )
      })}
    </div>
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

      <LangPicker
        value={activeLang}
        onChange={setActiveLang}
        options={langCounts}
        tgt={tgt}
        totalAll={totalAll}
      />

      <div className="ml-auto inline-flex items-center gap-2 text-xs text-text-subtle order-3">
        <span className="tabular shrink-0">{count} chương</span>
        <span className="text-border-soft">·</span>
        <SortToggle sortBy={sortBy} setSortBy={setSortBy} />
      </div>
    </div>
  )
}


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


function normalizeBcp(code: string | null): string {
  if (!code) return ''
  return code.toLowerCase().split(/[-_]/)[0]!
}


// useChapterSpawn — adapter over `useSpawnChapters` shaped for the
// work-route call site. Progress is keyed by chapter number (not
// version key), so the same slot tracks the spawn across the row's
// raw → translation transition. Each row reads its own progress via
// `getSpawnState(chapter.number)`; the full per-key map is also
// exposed so reactive watchers (e.g. the reader's "spawn done →
// toast" hook) can subscribe to phase transitions without polling.
//
// `workId` (optional) narrows the cache invalidation that fires when
// a spawn finishes, so a successful translate doesn't refetch every
// open work tab. Reader and work routes both know their work id and
// thread it through; older call sites can omit it and pay the broad-
// invalidate cost as before.
export function useChapterSpawn(targetLang: string | null, workId?: number) {
  const lang = targetLang ?? ''
  const ctl  = useSpawnChapters(lang)

  return {
    progressByKey: ctl.progressByKey,
    /** Read progress for a chapter (keyed by chapter.number). */
    getSpawnState: (chapterNumber: string) => ctl.getProgress(chapterNumber),
    /** Kick the pipeline for `chapter`, uploading from `raw`. */
    spawn: (chapter: HubChapter, raw: HubVersion) => {
      ctl.spawn(chapter.number, raw, chapter.label, workId)
    },
    abort:    (chapter: HubChapter) => ctl.abort(chapter.number),
    reset:    (chapter: HubChapter) => ctl.reset(chapter.number),
    resetAll: ctl.resetAll,
  }
}
