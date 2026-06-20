// /library — pinned works.
//
// A "library entry" is a Work with `in_library=true`. Browse-only works
// don't show up here; they live in /explore and the Recent rail.

import { useMemo, useState } from 'react'
import { createFileRoute, Link } from '@tanstack/react-router'
import { Plus, Search } from 'lucide-react'

import { useLibraryWorks, useLibraryStatusCounts } from '@features/library/queries'
import { WorkCard } from '@features/library/WorkCard'
import { LibraryStatusTabs, type LibraryStatusOrAll } from '@features/library/LibraryStatusTabs'
import { AddMangaModal } from '@features/library/AddMangaModal'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner, input as inputCls } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'


function LibraryPage() {
  const lib = useLibraryWorks()
  const counts = useLibraryStatusCounts()
  const [status, setStatus] = useState<LibraryStatusOrAll>('all')
  const [query,  setQuery]  = useState('')
  const [addOpen, setAddOpen] = useState(false)

  const filtered = useMemo(() => {
    const items = lib.data ?? []
    const byStatus = status === 'all'
      ? items
      : items.filter(it => it.library_status === status)
    const q = query.trim().toLowerCase()
    if (!q) return byStatus
    return byStatus.filter(it => it.title.toLowerCase().includes(q))
  }, [lib.data, status, query])

  if (lib.isPending) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner size={20} />
      </div>
    )
  }

  const items = lib.data ?? []
  if (items.length === 0) {
    return (
      <>
        <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10">
          <EmptyState
            title="Thư viện trống"
            hint="Tìm truyện hoặc dán URL để thêm vào."
            action={
              <Button variant="primary" size="md" onClick={() => setAddOpen(true)}>
                <Plus size={14} /> Thêm truyện
              </Button>
            }
          />
        </div>
        <AddMangaModal open={addOpen} onClose={() => setAddOpen(false)} />
      </>
    )
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-5">
      <header className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold text-text">Thư viện</h1>
            <p className="text-xs text-text-muted">{items.length} truyện</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="secondary" size="sm" onClick={() => setAddOpen(true)}>
              <Plus size={14} /> Thêm
            </Button>
            <Link to="/explore">
              <Button variant="ghost" size="sm">Khám phá</Button>
            </Link>
          </div>
        </div>

        <LibraryStatusTabs value={status} onChange={setStatus} counts={counts} />

        <div className="relative max-w-xl">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-subtle" />
          <input
            type="search"
            placeholder="Tìm trong thư viện…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            className={`${inputCls} pl-8`}
          />
        </div>
      </header>

      {filtered.length === 0 ? (
        <EmptyState
          title={query ? 'Không tìm thấy' : 'Trống ở mục này'}
          hint={query ? 'Thử từ khoá khác.' : 'Chuyển sang mục khác hoặc thêm truyện.'}
        />
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
          {filtered.map(it => (
            <WorkCard
              key={it.id}
              work={{
                id:        it.id,
                title:     it.title,
                cover_url: it.cover_url,
                source:    it.sources[0]?.source ?? null,
                nsfw:      it.nsfw,
              }}
            />
          ))}
        </div>
      )}

      <AddMangaModal open={addOpen} onClose={() => setAddOpen(false)} />
    </div>
  )
}

export const Route = createFileRoute('/library')({
  component: LibraryPage,
  staticData: { auth: 'required' },
})
