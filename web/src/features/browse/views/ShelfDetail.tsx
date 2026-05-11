import { useEffect, useMemo, useRef } from 'react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { ArrowLeft, AlertTriangle, Search } from 'lucide-react'
import { useHeaderStore } from '../../../store/header'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import {
  fetchBrowse, assembleFilterParams, getShelves, shelfPageSize,
} from '../manifest/runtime'
import { useAutoTranslate, shouldTranslate } from '../autoTranslate'
import { useTranslatedBatch } from '../useTranslated'
import type { InstalledSource, MangaSummary } from '../manifest/types'
import { FilterBar, type FilterState } from './FilterBar'
import { MangaCard } from './MangaCard'

// =============================================================================
// ShelfDetail — full-grid view of one shelf at /browse/$source/shelf/$shelfId.
//
// Pattern:
//   • Grid of MangaCard with infinite scroll
//   • FilterBar (if manifest declares filters that apply to this shelf)
//   • No search input (search has its own dedicated route)
// =============================================================================

interface Props {
  source:  InstalledSource
  shelfId: string
  filters: FilterState
  onFiltersChange: (next: FilterState) => void
}

export function ShelfDetail({ source, shelfId, filters, onFiltersChange }: Props) {
  const { manifest } = source
  const shelf = getShelves(manifest).find((s) => s.id === shelfId)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    if (shelf) {
      setHeader(shelf.label, [
        { label: 'Duyệt nguồn', to: '/browse' },
      ])
    }
    return () => clearHeader()
  }, [shelf, setHeader, clearHeader])

  const filterParams = useMemo(
    () => assembleFilterParams(manifest, filters),
    [manifest, filters],
  )

  const pageSize = shelfPageSize(manifest, shelfId)

  const {
    data, isError, error, isFetching, fetchNextPage, hasNextPage, isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey:         ['shelf', manifest.id, shelfId, filterParams, 'full'],
    initialPageParam: 1,
    queryFn: ({ pageParam }) =>
      fetchBrowse(manifest, shelfId, { page: pageParam, filterParams }),
    getNextPageParam: (last, all) => {
      if (last.length < pageSize) return undefined
      return all.length + 1
    },
    enabled:   !!shelf,
    staleTime: 60_000,
  })

  const all: MangaSummary[] = useMemo(
    () => (data?.pages ?? []).flat(),
    [data],
  )
  const showSkeleton = useDelayedFlag(!!shelf && all.length === 0 && isFetching, 250)

  // Auto-translate card titles
  const autoEnabled = useAutoTranslate((s) => s.enabled)
  const autoTarget  = useAutoTranslate((s) => s.target)
  const useTr = shouldTranslate(autoEnabled, autoTarget, manifest.languages)
  const titles = useMemo(() => all.map((m) => m.title), [all])
  const trTitles = useTranslatedBatch(titles, autoTarget, useTr)

  // Infinite scroll sentinel
  const sentinelRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!sentinelRef.current || !hasNextPage) return
    const io = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting && hasNextPage && !isFetchingNextPage) {
        void fetchNextPage()
      }
    }, { rootMargin: '600px' })
    io.observe(sentinelRef.current)
    return () => io.disconnect()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage])

  if (!shelf) {
    return (
      <div className="px-6 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Không tìm thấy mục này"
          hint="Nguồn không có mục này hoặc đã đổi tên."
        />
      </div>
    )
  }

  return (
    <div>
      {/* Mobile back */}
      <div className="sm:hidden px-4 pt-4">
        <Link
          to="/browse/$source"
          params={{ source: manifest.id }}
          className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text"
        >
          <ArrowLeft size={14} />
          {manifest.name}
        </Link>
      </div>

      <header className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4">
        <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-text">
          {shelf.label}
        </h1>
        <p className="text-sm text-text-subtle mt-1">{manifest.name}</p>
      </header>

      {manifest.filters && manifest.filters.length > 0 && (
        <div className="px-4 sm:px-6 mb-4">
          <FilterBar
            manifest={manifest}
            state={filters}
            onChange={onFiltersChange}
          />
        </div>
      )}

      <div className="px-4 sm:px-6 pb-8">
        {showSkeleton && <SkeletonGrid />}

        {isError && (
          <EmptyState
            icon={AlertTriangle}
            title="Không tải được"
            hint={(error as Error)?.message ?? 'Nguồn có thể đã đổi cấu trúc.'}
          />
        )}

        {!isFetching && !isError && all.length === 0 && !showSkeleton && (
          <EmptyState
            icon={Search}
            title="Không có kết quả"
            hint="Thử thay đổi bộ lọc hoặc xem mục khác."
          />
        )}

        {all.length > 0 && (
          <>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-x-3 gap-y-5">
              {all.map((m, i) => (
                <MangaCard
                  key={m.id}
                  source={manifest.id}
                  manifest={manifest}
                  manga={m}
                  translatedTitle={useTr ? trTitles[i] : null}
                />
              ))}
            </div>
            <div ref={sentinelRef} className="h-12 flex items-center justify-center mt-4">
              {isFetchingNextPage && <Spinner size={16} />}
              {!hasNextPage && (
                <span className="text-xs text-text-subtle">Hết kết quả</span>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function SkeletonGrid() {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-x-3 gap-y-5">
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={i} className="animate-pulse">
          <div className="w-full aspect-[2/3] rounded-md bg-surface-2 mb-2" />
          <div className="h-3 w-3/4 rounded bg-surface-2" />
        </div>
      ))}
    </div>
  )
}
