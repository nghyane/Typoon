import { useEffect, useMemo, useRef, useState } from 'react'
import { useInfiniteQuery } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { ArrowLeft, AlertTriangle, Search, X } from 'lucide-react'
import { useHeaderStore } from '../../../store/header'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner, input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { fetchBrowse, hasSearch, searchPageSize } from '../manifest/runtime'
import { useAutoTranslate, shouldTranslate } from '../autoTranslate'
import { useTranslatedBatch } from '../useTranslated'
import type { InstalledSource, MangaSummary } from '../manifest/types'
import { MangaCard } from './MangaCard'

// =============================================================================
// SearchResults — /browse/$source/search?q=...
//
// Same grid + infinite scroll pattern as ShelfDetail, with an editable
// search field at the top so the user can refine in place.
// =============================================================================

interface Props {
  source:    InstalledSource
  initialQ:  string
  onQueryChange: (q: string) => void
}

export function SearchResults({ source, initialQ, onQueryChange }: Props) {
  const { manifest } = source
  const [q, setQ] = useState(initialQ)
  useEffect(() => { setQ(initialQ) }, [initialQ])

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Tìm kiếm', [
      { label: manifest.name, to: '/browse/$source' },
    ])
    return () => clearHeader()
  }, [manifest.name, setHeader, clearHeader])

  const searchable = hasSearch(manifest)
  const trimmed    = initialQ.trim()
  const pageSize   = searchPageSize(manifest)

  const {
    data, isError, error, isFetching, fetchNextPage, hasNextPage, isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey:         ['search', manifest.id, trimmed],
    initialPageParam: 1,
    queryFn: ({ pageParam }) =>
      fetchBrowse(manifest, { search: true }, { page: pageParam, q: trimmed }),
    getNextPageParam: (last, all) => {
      if (last.length < pageSize) return undefined
      if (last.length < pageSize) return undefined
      return all.length + 1
    },
    enabled:   searchable && trimmed.length > 0,
    staleTime: 60_000,
  })

  const all: MangaSummary[] = useMemo(
    () => (data?.pages ?? []).flat(),
    [data],
  )
  const showSkeleton = useDelayedFlag(
    trimmed.length > 0 && all.length === 0 && isFetching, 250,
  )

  const autoEnabled = useAutoTranslate((s) => s.enabled)
  const autoTarget  = useAutoTranslate((s) => s.target)
  const useTr = shouldTranslate(autoEnabled, autoTarget, manifest.languages)
  const titles = useMemo(() => all.map((m) => m.title), [all])
  const trTitles = useTranslatedBatch(titles, autoTarget, useTr)

  // Infinite scroll
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

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onQueryChange(q.trim())
  }

  return (
    <div>
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

      <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4">
        <form onSubmit={onSubmit}>
          <label className={cn(inputCls, 'flex items-center gap-2 max-w-xl cursor-text h-10')}>
            <Search size={14} className="text-text-subtle shrink-0" />
            <input
              type="text"
              placeholder={`Tìm trên ${manifest.name}…`}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              autoFocus
              className="flex-1 bg-transparent outline-none text-sm placeholder:text-text-subtle text-text min-w-0"
            />
            {q && (
              <button
                type="button"
                onClick={() => { setQ(''); onQueryChange('') }}
                className="text-text-subtle hover:text-text cursor-pointer"
                title="Xoá"
                aria-label="Xoá tìm kiếm"
              >
                <X size={14} />
              </button>
            )}
          </label>
        </form>
        {trimmed && (
          <p className="text-xs text-text-subtle mt-2">
            Kết quả cho <span className="text-text font-medium">"{trimmed}"</span>
          </p>
        )}
      </div>

      <div className="px-4 sm:px-6 pb-8">
        {!searchable ? (
          <EmptyState
            icon={Search}
            title="Nguồn không hỗ trợ tìm kiếm"
            hint="Hãy duyệt qua mục Mới cập nhật / Phổ biến."
          />
        ) : !trimmed ? (
          <EmptyState
            icon={Search}
            title="Nhập từ khoá để tìm"
            hint="Tên truyện, tác giả hoặc một phần tựa đề."
          />
        ) : showSkeleton ? (
          <SkeletonGrid />
        ) : isError ? (
          <EmptyState
            icon={AlertTriangle}
            title="Không tải được"
            hint={(error as Error)?.message ?? 'Thử lại sau.'}
          />
        ) : all.length === 0 ? (
          <EmptyState
            icon={Search}
            title="Không có kết quả"
            hint={`Không tìm thấy truyện nào khớp "${trimmed}".`}
          />
        ) : (
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
