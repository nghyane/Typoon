// /explore — single-page source browser.
//
// Structure:
//   ExplorePage
//     SourceSession (key=activeId — unmount/remount khi đổi nguồn,
//                   reset state tự động, không cần useEffect)
//       ├── sticky bar: source tabs + shelf tabs + search + filters
//       └── manga grid (ShelfContent | SearchContent)
//
// Context pattern: SourceSession bọc Context.Provider. Cả sticky bar
// lẫn grid đều là children → context reach được mà không prop-drill.

import {
  createContext, useContext, useRef, useState, useEffect,
  useCallback, useMemo, type ReactNode,
} from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { Search, X, Loader2, AlertCircle, Compass, Settings } from 'lucide-react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'

import { cn } from '@shared/lib/cn'
import { useDebouncedValue } from '@shared/lib/useDebouncedValue'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { Cover } from '@shared/ui/Cover'
import { useEnabledSources } from '@features/browse/sources'
import {
  getShelves, getFilters, getDefaultFilterState,
  assembleFilterParams, assembleFilterState, fetchBrowse, hasSearch, searchPageSize,
} from '@features/browse/manifest/runtime'
import { useSession as useAuthSession } from '@features/auth/session'
import { importMaterialFromHit } from '@features/material/import'
import { toast } from '@shared/ui/Toaster'
import type { InstalledSource, MangaSummary } from '@features/browse/manifest/types'
import { useShelfQuery } from '@features/explore/useShelfQuery'
import { FilterChips } from '@features/explore/FilterChips'
import { useHeaderStore } from '../store/header'


// ─── accent ────────────────────────────────────────────────────────

const DOT: Record<string, string> = {
  orange:  'bg-orange-400',  emerald: 'bg-emerald-400',
  rose:    'bg-rose-400',    blue:    'bg-blue-400',
  violet:  'bg-violet-400',  amber:   'bg-amber-400',
  cyan:    'bg-cyan-400',    sky:     'bg-sky-400',
}
const dot = (accent?: string) => DOT[accent ?? ''] ?? 'bg-text-subtle/40'


// ─── session context ───────────────────────────────────────────────
// State cho một nguồn. SourceSession mount mới khi sourceId thay đổi
// (key prop) nên state reset tự động — không cần useEffect cleanup.

interface SessionCtx {
  source:      InstalledSource
  shelf:       string
  setShelf:    (id: string) => void
  filterState: Record<string, string | string[]>
  setFilter:   (s: Record<string, string | string[]>) => void
  rawQuery:    string
  setRawQuery: (v: string) => void
  query:       string   // debounced
  inputRef:    React.RefObject<HTMLInputElement | null>
  pending:     string | null
  pick:        (manga: MangaSummary) => void
}

const Ctx = createContext<SessionCtx | null>(null)
const useSession = () => {
  const c = useContext(Ctx)
  if (!c) throw new Error('outside SourceSession')
  return c
}

function SourceSession({ source, children }: { source: InstalledSource; children: ReactNode }) {
  const nav      = useNavigate()
  const shelves  = getShelves(source.manifest)
  const canSearch = hasSearch(source.manifest)

  const [shelf,       setShelf]    = useState(shelves[0]?.id ?? '')
  const [filterState, setFilter]   = useState(() => getDefaultFilterState(source.manifest))
  const [rawQuery,    setRawQuery] = useState('')
  const [pending,     setPending]  = useState<string | null>(null)
  const inputRef                   = useRef<HTMLInputElement>(null)
  const query                      = useDebouncedValue(rawQuery, 250)

  const pick = useCallback(async (manga: MangaSummary) => {
    if (pending) return
    setPending(manga.id)
    try {
      const mat = await importMaterialFromHit({ manga, source, score: 0 })
      nav({ to: '/w/$workId', params: { workId: String(mat.work_id) } })
    } catch (e) {
      toast.error((e as Error).message ?? 'Không thể mở manga này')
      setPending(null)
    }
  }, [pending, source, nav])

  // Inject search input vào Header slot khi source hỗ trợ search.
  // Clear slot khi SourceSession unmount (đổi source hoặc rời route).
  const { setSlot } = useHeaderStore()
  useEffect(() => {
    if (!canSearch) return
    setSlot(
      <SearchInput
        source={source}
        rawQuery={rawQuery}
        query={query}
        inputRef={inputRef}
        onChangeQuery={setRawQuery}
      />
    )
    return () => setSlot(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canSearch, rawQuery, query, source])

  return (
    <Ctx.Provider value={{
      source, shelf, setShelf, filterState, setFilter,
      rawQuery, setRawQuery, query, inputRef, pending, pick,
    }}>
      {children}
    </Ctx.Provider>
  )
}


// ─── page ──────────────────────────────────────────────────────────

function ExplorePage() {
  const nav     = useNavigate()
  const { user } = useAuthSession()
  const prefLang = user?.preferred_target_lang ?? 'vi'

  const raw = useEnabledSources().filter((s) => s.manifest.kind !== 'internal')

  // Nguồn có preferred lang → lên đầu, giữ thứ tự tương đối bên trong mỗi nhóm
  const sources = [...raw].sort((a, b) => {
    const aHas = a.manifest.languages.includes(prefLang) ? 0 : 1
    const bHas = b.manifest.languages.includes(prefLang) ? 0 : 1
    return aHas - bHas
  })

  const [activeId, setActiveId] = useState(() => sources[0]?.manifest.id ?? '')

  // Nếu source bị remove trong khi đang active, fall back về first
  const active = sources.find((s) => s.manifest.id === activeId)
    ?? sources[0]
    ?? null

  if (sources.length === 0) {
    return (
      <div className="px-4 sm:px-6 py-16 max-w-sm mx-auto">
        <EmptyState
          icon={Compass}
          title="Chưa có nguồn nào"
          hint="Bật thêm nguồn để bắt đầu khám phá manga."
          action={
            <Button variant="secondary" size="sm" onClick={() => nav({ to: '/settings' })}>
              <Settings size={13} />
              Quản lý nguồn
            </Button>
          }
        />
      </div>
    )
  }

  return (
    // key=active.manifest.id: SourceSession unmount + remount khi đổi nguồn
    // → state reset tự động, không cần useEffect
    <SourceSession key={active!.manifest.id} source={active!}>
      <div>
        {/* sticky bar */}
        <div className="sticky top-0 z-10 bg-bg">
          <TopBar
            sources={sources}
            activeId={active!.manifest.id}
            onSelect={(id) => setActiveId(id)}
          />
        </div>

        {/* grid */}
        <div className="px-4 sm:px-6 py-4">
          <SourceGrid />
        </div>
      </div>
    </SourceSession>
  )
}


// ─── TopBar ────────────────────────────────────────────────────────
// Source chips + shelf chips + filter chips.
// Search input đã được inject vào Header slot — không cần hàng riêng.

function TopBar({
  sources, activeId, onSelect,
}: {
  sources:  InstalledSource[]
  activeId: string
  onSelect: (id: string) => void
}) {
  const {
    source, shelf, setShelf, rawQuery, filterState, setFilter,
  } = useSession()

  const shelves  = getShelves(source.manifest)
  const filters  = getFilters(source.manifest)
  const isSearch = rawQuery.trim().length > 0

  const hasRow2 = shelves.length > 1 || filters.length > 0

  return (
    <div className="px-4 sm:px-6 pt-2 pb-1">

      {/* hàng 1 — source chips */}
      <div
        className="flex items-center gap-1.5 overflow-x-auto"
        style={{ scrollbarWidth: 'none' }}
      >
        {sources.map((s) => {
          const active_ = s.manifest.id === activeId
          return (
            <button
              key={s.manifest.id}
              type="button"
              onClick={() => onSelect(s.manifest.id)}
              className={cn(
                'inline-flex items-center gap-1.5 h-8 px-3 rounded-full text-sm font-medium',
                'whitespace-nowrap shrink-0 transition-colors duration-150 cursor-pointer',
                active_
                  ? 'bg-surface-2 text-text'
                  : 'text-text-muted hover:text-text',
              )}
            >
              <span className={cn(
                'size-1.5 rounded-full shrink-0',
                dot(s.manifest.accent),
                !active_ && 'opacity-50',
              )} />
              {s.manifest.name}
            </button>
          )
        })}
      </div>

      {/* hàng 2 — shelf chips · filter chips (ẩn khi đang search) */}
      {hasRow2 && !isSearch && (
        <div
          className="flex items-center gap-1.5 overflow-x-auto mt-2"
          style={{ scrollbarWidth: 'none' }}
        >
          {shelves.length > 1 && shelves.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setShelf(s.id)}
              className={cn(
                'inline-flex items-center h-8 px-3 rounded-full text-sm font-medium',
                'whitespace-nowrap shrink-0 transition-colors duration-150 cursor-pointer',
                shelf === s.id
                  ? 'bg-surface-2 text-text'
                  : 'text-text-muted hover:text-text',
              )}
            >
              {s.label}
            </button>
          ))}

          {shelves.length > 1 && filters.length > 0 && (
            <span className="w-px h-4 bg-border-soft shrink-0" />
          )}

          {filters.length > 0 && (
            <FilterChips filters={filters} state={filterState} onChange={setFilter} />
          )}
        </div>
      )}
    </div>
  )
}


// ─── SearchInput ───────────────────────────────────────────────────
// Rendered inside the Header slot — shared between mobile + desktop.

function SearchInput({
  source, rawQuery, query, inputRef, onChangeQuery,
}: {
  source:        InstalledSource
  rawQuery:      string
  query:         string
  inputRef:      React.RefObject<HTMLInputElement | null>
  onChangeQuery: (v: string) => void
}) {
  const isSearch  = rawQuery.trim().length > 0
  const isPending = isSearch && rawQuery !== query

  const clear = useCallback(() => {
    onChangeQuery('')
    inputRef.current?.focus()
  }, [onChangeQuery, inputRef])

  return (
    <div className="flex items-center gap-2 h-8 px-3 rounded-md bg-surface-2 w-full">
      {isPending
        ? <Loader2 size={13} className="shrink-0 text-text-muted animate-spin" />
        : <Search  size={13} className="shrink-0 text-text-muted" />
      }
      <input
        ref={inputRef}
        type="search"
        value={rawQuery}
        onChange={(e) => onChangeQuery(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Escape') clear() }}
        placeholder={`Tìm trên ${source.manifest.name}…`}
        className="flex-1 min-w-0 bg-transparent text-sm text-text placeholder:text-text-muted outline-none"
      />
      {isSearch && (
        <button
          type="button"
          onClick={clear}
          className="size-5 flex items-center justify-center rounded-sm text-text-muted hover:text-text transition-colors cursor-pointer shrink-0"
        >
          <X size={11} />
        </button>
      )}
    </div>
  )
}


// ─── SourceGrid ────────────────────────────────────────────────────

function SourceGrid() {
  const { source, shelf, filterState, query, pending, pick } = useSession()
  const filterParams = assembleFilterParams(source.manifest, filterState)
  const filterSt     = assembleFilterState(source.manifest, filterState)
  const isSearch     = query.trim().length > 0

  return isSearch
    ? <SearchContent source={source} query={query}   filterParams={filterParams} filterState={filterSt} pending={pending} onPick={pick} />
    : <ShelfContent  source={source} shelfId={shelf} filterParams={filterParams} filterState={filterSt} pending={pending} onPick={pick} />
}


// ─── useScrollEl — find the AppLayout overflow-auto ancestor ───────
// Mirrors WorkChapterList: AppLayout uses <main overflow-auto>, not
// window. Must find it by ancestor walk so the virtualizer scrolls
// the right element.

function useScrollEl(ref: React.RefObject<HTMLElement | null>) {
  const [scrollEl, setScrollEl] = useState<HTMLElement | null>(null)
  const [scrollMargin, setScrollMargin] = useState(0)

  useEffect(() => {
    let el: HTMLElement | null = ref.current
    while (el && el !== document.body) {
      const oy = getComputedStyle(el).overflowY
      if (oy === 'auto' || oy === 'scroll') { setScrollEl(el); break }
      el = el.parentElement
    }
    if (!el || el === document.body) {
      setScrollEl((document.scrollingElement as HTMLElement) ?? null)
    }
  }, [ref])

  useEffect(() => {
    if (!scrollEl || !ref.current) return
    const measure = () => {
      if (!ref.current || !scrollEl) return
      const listTop   = ref.current.getBoundingClientRect().top
      const scrollTop = scrollEl.getBoundingClientRect().top
      setScrollMargin(listTop - scrollTop + scrollEl.scrollTop)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(scrollEl)
    if (ref.current.parentElement) ro.observe(ref.current.parentElement)
    return () => ro.disconnect()
  }, [scrollEl, ref])

  return { scrollEl, scrollMargin }
}


// ─── COLS — responsive column count (mirrors CSS grid breakpoints) ──
// Used to convert flat item index → row index for the virtualizer.

const BREAKPOINTS: [number, number][] = [
  [1280, 7], // xl
  [1024, 6], // lg
  [768,  5], // md
  [640,  4], // sm
  [0,    3], // base
]

function useCols(containerRef: React.RefObject<HTMLElement | null>): number {
  const [cols, setCols] = useState(3)
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const update = () => {
      const w = el.getBoundingClientRect().width
      const c = BREAKPOINTS.find(([bp]) => w >= bp)?.[1] ?? 3
      setCols(c)
    }
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [containerRef])
  return cols
}


// ─── VirtualGrid ────────────────────────────────────────────────────
// Shared virtualised grid used by ShelfContent + SearchContent.
// items      — flat array of MangaSummary
// sentinel   — when true, the last virtual row includes a sentinel slot
//              that triggers fetchNextPage when scrolled into view.

function VirtualGrid({
  items,
  pending,
  onPick,
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage,
}: {
  items:              MangaSummary[]
  pending:            string | null
  onPick:             (m: MangaSummary) => void
  hasNextPage:        boolean
  isFetchingNextPage: boolean
  fetchNextPage:      () => void
}) {
  const outerRef    = useRef<HTMLDivElement>(null)
  const { scrollEl, scrollMargin } = useScrollEl(outerRef)
  const cols        = useCols(outerRef)

  // Flat items → rows of `cols` length. Last row may be shorter.
  const rows = useMemo(() => {
    const r: MangaSummary[][] = []
    for (let i = 0; i < items.length; i += cols) r.push(items.slice(i, i + cols))
    return r
  }, [items, cols])

  // Sentinel row: appended when more pages exist.
  const totalRows   = rows.length + (hasNextPage ? 1 : 0)
  const sentinelRow = rows.length  // index of the sentinel (may not exist)

  // Row height estimate: cover (aspect-2/3 of ~(containerWidth/cols)) +
  // two text lines + gap. We use a fixed 260 px as a safe over-estimate;
  // the virtualizer measures after paint and self-corrects.
  const virt = useVirtualizer({
    count:            totalRows,
    getScrollElement: () => scrollEl,
    estimateSize:     () => 260,
    overscan:         3,
    scrollMargin,
  })

  // Trigger next-page fetch when sentinel row enters the virtualizer's
  // rendered window.
  const virtualItems = virt.getVirtualItems()
  const lastVi       = virtualItems[virtualItems.length - 1]
  useEffect(() => {
    if (!hasNextPage || isFetchingNextPage) return
    if (lastVi && lastVi.index >= sentinelRow) fetchNextPage()
  }, [lastVi, hasNextPage, isFetchingNextPage, fetchNextPage, sentinelRow])

  return (
    <div
      ref={outerRef}
      className="relative w-full"
      style={{ height: virt.getTotalSize() }}
    >
      {virtualItems.map((vi) => {
        const isSentinel = vi.index === sentinelRow && hasNextPage
        const row        = rows[vi.index]

        return (
          <div
            key={vi.key}
            data-index={vi.index}
            ref={virt.measureElement}
            className="absolute left-0 right-0"
            style={{ transform: `translateY(${vi.start - virt.options.scrollMargin}px)` }}
          >
            {isSentinel ? (
              // Sentinel: spinner row that triggers next-page load
              <div className="flex justify-center items-center py-8">
                <Loader2 size={18} className="text-text-muted animate-spin" />
              </div>
            ) : row ? (
              <div
                className="grid gap-3 sm:gap-4 pb-3 sm:pb-4"
                style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
              >
                {row.map((m) => (
                  <Card
                    key={m.id}
                    manga={m}
                    pending={pending === m.id}
                    onClick={() => onPick(m)}
                  />
                ))}
                {/* fill phantom cells so last row aligns left */}
                {row.length < cols && Array.from({ length: cols - row.length }).map((_, i) => (
                  <div key={`ph-${i}`} aria-hidden />
                ))}
              </div>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}


// ─── ShelfContent ───────────────────────────────────────────────────

function ShelfContent({ source, shelfId, filterParams, filterState, pending, onPick }: {
  source:       InstalledSource
  shelfId:      string
  filterParams: string
  filterState:  Record<string, string | string[]>
  pending:      string | null
  onPick:       (m: MangaSummary) => void
}) {
  const {
    items, loading, error,
    hasNextPage, isFetchingNextPage, fetchNextPage,
  } = useShelfQuery(source, shelfId, filterParams, filterState)

  if (loading) return <GridSkeleton />
  if (error)   return <Err msg={error.message} />
  if (!items.length) return <Empty msg="Không có dữ liệu." />

  return (
    <VirtualGrid
      items={items}
      pending={pending}
      onPick={onPick}
      hasNextPage={hasNextPage}
      isFetchingNextPage={isFetchingNextPage}
      fetchNextPage={fetchNextPage}
    />
  )
}


// ─── SearchContent ──────────────────────────────────────────────────

function SearchContent({ source, query, filterParams, filterState, pending, onPick }: {
  source:       InstalledSource
  query:        string
  filterParams: string
  filterState:  Record<string, string | string[]>
  pending:      string | null
  onPick:       (m: MangaSummary) => void
}) {
  const q         = query.trim()
  const enabled   = q.length >= 2
  const pageSize  = searchPageSize(source.manifest)
  const paginated = pageSize !== Infinity

  const {
    data, isPending, isFetchingNextPage, hasNextPage, fetchNextPage, error,
  } = useInfiniteQuery({
    queryKey:         ['explore', 'search', source.manifest.id, q, filterParams],
    queryFn:          ({ pageParam }) =>
      fetchBrowse(source.manifest, { search: true }, { q, page: pageParam as number, filterParams, filterState }),
    initialPageParam: 1,
    getNextPageParam: (last, _all, lastParam) => {
      if (!paginated) return undefined
      if (last.length < pageSize) return undefined
      return (lastParam as number) + 1
    },
    staleTime: 5 * 60_000,
    retry:     false,
    enabled,
  })

  const items = data?.pages.flat() ?? []

  if (!enabled)                   return null
  if (isPending && !items.length) return <GridSkeleton />
  if (error)                      return <Err msg={(error as Error).message} />
  if (!items.length)              return <Empty msg={`Không tìm thấy "${q}"`} />

  return (
    <VirtualGrid
      items={items}
      pending={pending}
      onPick={onPick}
      hasNextPage={hasNextPage ?? false}
      isFetchingNextPage={isFetchingNextPage}
      fetchNextPage={fetchNextPage}
    />
  )
}


// ─── Card ───────────────────────────────────────────────────────────

function Card({ manga, pending, onClick }: {
  manga:   MangaSummary
  pending: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={pending}
      className="group text-left flex flex-col gap-1.5 cursor-pointer disabled:cursor-wait"
    >
      <div className="relative w-full aspect-[2/3] rounded-md overflow-hidden bg-surface-2">
        <Cover
          src={manga.cover}
          title={manga.title}
          className="absolute inset-0 group-hover:brightness-110 transition-[filter] duration-200"
        />
        {pending && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
            <Loader2 size={18} className="text-white animate-spin" />
          </div>
        )}
      </div>
      <p className="text-xs font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors duration-150">
        {manga.title}
      </p>
    </button>
  )
}


// ─── primitives ─────────────────────────────────────────────────────

function GridSkeleton() {
  return (
    <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 xl:grid-cols-7 gap-3 sm:gap-4">
      {Array.from({ length: 21 }).map((_, i) => (
        <div key={i} className="flex flex-col gap-1.5 animate-pulse">
          <div className="w-full aspect-[2/3] rounded-md bg-surface-2" />
          <div className="h-2 w-4/5 rounded bg-surface-2" />
          <div className="h-2 w-3/5 rounded bg-surface-2" />
        </div>
      ))}
    </div>
  )
}

function Err({ msg }: { msg: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-error py-10">
      <AlertCircle size={14} /> {msg}
    </div>
  )
}

function Empty({ msg }: { msg: string }) {
  return <p className="text-sm text-text-muted text-center py-10">{msg}</p>
}


// ─── route ─────────────────────────────────────────────────────────

export const Route = createFileRoute('/explore')({
  component: ExplorePage,
})
