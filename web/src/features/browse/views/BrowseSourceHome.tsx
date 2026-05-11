import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowLeft, Search } from 'lucide-react'
import { Link, useNavigate } from '@tanstack/react-router'
import { useHeaderStore } from '../../../store/header'
import { input as inputCls } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import {
  fetchBrowse, getShelves, hasSearch, type ShelfDescriptor,
} from '../manifest/runtime'
import { useSourceContinueRail } from '@features/library/hooks'
import { LibraryRailCard } from '@features/library/views/LibraryCard'
import type { InstalledSource } from '../manifest/types'
import { Shelf } from './Shelf'
import { ShelfCard } from './ShelfCard'

// =============================================================================
// BrowseSourceHome — landing for /browse/$source.
//
// View is source-kind-agnostic: it asks `runtime.getShelves(manifest)`
// for what to render and `runtime.fetchBrowse(manifest, shelfId)` for
// the data. External manifests (HappyMH, MangaDex, …) and internal
// sources (Community) reach this same component the same way.
// =============================================================================

const SHELF_PEEK = 12

interface Props { source: InstalledSource }

export function BrowseSourceHome({ source }: Props) {
  const { manifest } = source
  const [q, setQ] = useState('')
  const nav = useNavigate()
  const continueItems = useSourceContinueRail(manifest.id, 8)

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader(manifest.name, [{ label: 'Duyệt nguồn', to: '/browse' }])
    return () => clearHeader()
  }, [manifest.name, setHeader, clearHeader])

  const shelves     = getShelves(manifest)
  const searchable  = hasSearch(manifest)

  const onSubmitSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!q.trim()) return
    nav({
      to: '/browse/$source/search',
      params: { source: manifest.id },
      search: { q: q.trim() } as never,
    })
  }

  return (
    <div>
      {/* Mobile back */}
      <div className="sm:hidden px-4 pt-4">
        <Link
          to="/browse"
          className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text"
        >
          <ArrowLeft size={14} />
          Duyệt nguồn
        </Link>
      </div>

      {searchable && (
        <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4">
          <form onSubmit={onSubmitSearch}>
            <label className={cn(inputCls, 'flex items-center gap-2 max-w-xl cursor-text h-10')}>
              <Search size={14} className="text-text-subtle shrink-0" />
              <input
                type="text"
                placeholder={`Tìm trên ${manifest.name}…`}
                value={q}
                onChange={(e) => setQ(e.target.value)}
                className="flex-1 bg-transparent outline-none text-sm placeholder:text-text-subtle text-text min-w-0"
              />
            </label>
          </form>
        </div>
      )}

      {continueItems.length > 0 && (
        <Shelf label="Tiếp tục đọc">
          {continueItems.map((entry) => (
            <LibraryRailCard key={entry.mangaUrl} entry={entry} />
          ))}
        </Shelf>
      )}

      {shelves.map((shelf) => (
        <ShelfRail
          key={shelf.id}
          source={source}
          shelf={shelf}
        />
      ))}

      <div className="h-16" />
    </div>
  )
}

function ShelfRail({
  source, shelf,
}: {
  source: InstalledSource
  shelf:  ShelfDescriptor
}) {
  const { manifest } = source
  const { data, isPending } = useQuery({
    queryKey:  ['shelf', manifest.id, shelf.id, 'preview'],
    queryFn:   () => fetchBrowse(manifest, shelf.id, { page: 1 }),
    staleTime: 5 * 60_000,
  })

  const items = (data ?? []).slice(0, SHELF_PEEK)
  const hasMore = shelf.paginated || (data?.length ?? 0) > SHELF_PEEK

  return (
    <Shelf
      label={shelf.label}
      hint={shelf.hint}
      skeleton={isPending}
      more={hasMore ? {
        to: '/browse/$source/shelf/$shelfId',
        params: { source: manifest.id, shelfId: shelf.id },
      } : undefined}
    >
      {items.map((m) => (
        <ShelfCard
          key={m.id}
          source={manifest.id}
          manifest={manifest}
          manga={m}
        />
      ))}
    </Shelf>
  )
}
