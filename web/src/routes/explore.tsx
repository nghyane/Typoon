// /explore — multi-source search.
//
// Tab per enabled source. Click a result → resolve-or-create the Work
// (browse-only, not pinned) and navigate to /w/$workId. Adding to
// library is a separate, explicit action on the Work hub.

import { useEffect, useState } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { Search, Compass, Plus, Check } from 'lucide-react'
import { useQuery } from '@tanstack/react-query'

import { useEnabledSources } from '@features/browse/sources'
import {
  fetchBrowse, getShelves, hasSearch,
} from '@features/browse/manifest/runtime'
import type { MangaSummary } from '@features/browse/manifest/types'
import { Cover } from '@shared/ui/Cover'
import { useEnsureWorkFromSource, useWorkBySourceRef } from '@features/works/queries'
import { AddMangaModal } from '@features/library/AddMangaModal'
import { useDebouncedValue } from '@shared/lib/useDebouncedValue'
import { qk } from '@shared/api/keys'
import { Spinner, Tag, input as inputCls } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { toast } from '@shared/ui/Toaster'
import { cn } from '@shared/lib/cn'


function ExplorePage() {
  const sources = useEnabledSources()
  const [activeId, setActiveId] = useState<string | null>(sources[0]?.manifest.id ?? null)
  const [query, setQuery]       = useState('')
  const [addOpen, setAddOpen]   = useState(false)
  const debouncedQuery          = useDebouncedValue(query, 400)

  useEffect(() => {
    if (!activeId && sources[0]) setActiveId(sources[0].manifest.id)
  }, [sources, activeId])

  const active = sources.find(s => s.manifest.id === activeId)

  if (sources.length === 0) {
    return (
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10">
        <EmptyState
          title="Chưa có nguồn nào"
          hint="Mở Settings → Nguồn để bật ít nhất một nguồn."
        />
      </div>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">
      <header className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <h1 className="text-lg font-semibold text-text inline-flex items-center gap-2">
            <Compass size={18} /> Khám phá
          </h1>
          <Button variant="secondary" size="sm" onClick={() => setAddOpen(true)}>
            <Plus size={14} /> Thêm
          </Button>
        </div>

        <div className="flex flex-wrap gap-2" role="tablist">
          {sources.map(s => {
            const sel = s.manifest.id === activeId
            return (
              <button
                key={s.manifest.id}
                type="button"
                role="tab"
                aria-selected={sel}
                onClick={() => setActiveId(s.manifest.id)}
                className={cn(
                  'inline-flex items-center gap-2 h-7 px-3 rounded-full text-xs font-medium transition-colors',
                  sel
                    ? 'bg-accent-bg text-accent-text'
                    : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
                )}
              >
                {s.manifest.name}
              </button>
            )
          })}
        </div>

        {active && hasSearch(active.manifest) && (
          <div className="relative max-w-xl">
            <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
            <input
              type="search"
              placeholder={`Tìm trong ${active.manifest.name}…`}
              value={query}
              onChange={e => setQuery(e.target.value)}
              className={`${inputCls} pl-8`}
            />
          </div>
        )}
      </header>

      {active && (
        <SourceContent source={active} query={debouncedQuery.trim()} />
      )}

      <AddMangaModal open={addOpen} onClose={() => setAddOpen(false)} />
    </div>
  )
}


// ── Per-source content ──────────────────────────────────────────────

function SourceContent({
  source, query,
}: {
  source: ReturnType<typeof useEnabledSources>[number]
  query:  string
}) {
  const manifest = source.manifest
  const shelves  = getShelves(manifest)
  const [shelfId, setShelfId] = useState<string>(shelves[0]?.id ?? '')

  useEffect(() => {
    if (!shelfId && shelves[0]) setShelfId(shelves[0].id)
  }, [shelves, shelfId])

  const target = query
    ? { search: true as const }
    : shelfId

  const browse = useQuery<MangaSummary[]>({
    queryKey: query
      ? qk.manifest.search(manifest.id, query)
      : ['manifest', 'shelf', manifest.id, shelfId, 1],
    queryFn:  () =>
      fetchBrowse(manifest, target, { page: 1, q: query || undefined }),
    enabled:  query.length > 0 || !!shelfId,
    staleTime: 5 * 60_000,
  })

  return (
    <section className="space-y-4">
      {!query && shelves.length > 1 && (
        <div className="flex flex-wrap gap-2">
          {shelves.map(s => {
            const sel = s.id === shelfId
            return (
              <button
                key={s.id}
                type="button"
                onClick={() => setShelfId(s.id)}
                className={cn(
                  'h-7 px-3 rounded-full text-xs font-medium transition-colors',
                  sel
                    ? 'bg-surface-2 text-text'
                    : 'text-text-muted hover:text-text',
                )}
              >
                {s.label}
              </button>
            )
          })}
        </div>
      )}

      {browse.isPending ? (
        <div className="flex items-center justify-center py-12"><Spinner size={20} /></div>
      ) : browse.error ? (
        <EmptyState title="Không tải được" hint={browse.error.message} />
      ) : (browse.data?.length ?? 0) === 0 ? (
        <EmptyState
          title={query ? 'Không tìm thấy' : 'Chưa có truyện ở đây'}
          hint={query ? 'Thử từ khoá khác.' : undefined}
        />
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
          {browse.data!.map(m => (
            <BrowseCard key={m.id} source={source} manga={m} />
          ))}
        </div>
      )}
    </section>
  )
}

// Card that resolves the manga to a Work on click. Browse-only by
// default — adding to library is a separate action on the Work hub.
function BrowseCard({
  source, manga,
}: {
  source: ReturnType<typeof useEnabledSources>[number]
  manga:  MangaSummary
}) {
  const nav     = useNavigate()
  const ensure  = useEnsureWorkFromSource()
  const lookup  = useWorkBySourceRef(source.manifest.id, manga.url)

  const existing  = lookup.data ?? null
  const inLibrary = !!existing?.in_library

  async function handleClick() {
    try {
      const work = await ensure.mutateAsync({
        source:       source.manifest.id,
        upstream_ref: manga.url,
        snapshot: {
          title:       manga.title,
          cover_url:   manga.cover ?? null,
          source_lang: source.manifest.languages?.[0] ?? 'ja',
          target_lang: 'vi',
          nsfw:        !!source.manifest.nsfw,
          languages:   source.manifest.languages ?? [],
        },
      })
      nav({ to: '/w/$workId', params: { workId: work.id }, search: { tab: undefined } })
    } catch (e) {
      toast.error((e as Error).message)
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={ensure.isPending}
      className="group flex flex-col gap-2 rounded-sm overflow-hidden text-left disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
    >
      <div className="relative aspect-[3/4] rounded-sm overflow-hidden bg-surface-2">
        <Cover
          src={manga.cover}
          title={manga.title}
          className="absolute inset-0 transition-transform group-hover:scale-[1.02]"
        />
        {source.manifest.nsfw && (
          <Tag tone="error" size="sm" className="absolute top-1.5 right-1.5">18+</Tag>
        )}
      </div>
      <div className="px-0.5 space-y-0.5">
        <div className="text-xs font-medium text-text line-clamp-2 leading-tight">
          {manga.title}
        </div>
        <div className="flex items-center gap-1.5 text-xs uppercase tracking-wider text-text-subtle">
          <span className="truncate">{source.manifest.name}</span>
          {inLibrary && (
            <Check
              size={12}
              className="shrink-0 text-success"
              aria-label="Trong thư viện"
            />
          )}
        </div>
      </div>
    </button>
  )
}

export const Route = createFileRoute('/explore')({
  component: ExplorePage,
  staticData: { auth: 'required' },
})
