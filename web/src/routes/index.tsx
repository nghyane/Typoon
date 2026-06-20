// Home — reader-first landing page.

import { createFileRoute, Link } from '@tanstack/react-router'
import { BookOpen, Library, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

import { useSession } from '@features/auth/session'
import { useLibraryWorks } from '@features/library/queries'
import { WorkCard } from '@features/library/WorkCard'
import { useRecentlyOpened } from '@features/works/queries'
import { Button } from '@shared/ui/Button'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'


function HomePage() {
  const session = useSession()
  const recent  = useRecentlyOpened(6)
  const library = useLibraryWorks()

  if (session.status === 'loading') return null

  const recentWorks  = recent.data ?? []
  const libraryWorks = library.data ?? []
  const loading = recent.isPending || library.isPending

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-8">
      {loading ? (
        <div className="py-12 flex justify-center"><Spinner size={20} /></div>
      ) : (
        <>
          <WorkRail
            title="Đọc tiếp"
            icon={BookOpen}
            works={recentWorks}
            emptyTitle="Chưa mở truyện nào"
            emptyHint="Khám phá truyện để bắt đầu."
          />

          <WorkRail
            title="Thư viện"
            icon={Library}
            works={libraryWorks.slice(0, 6)}
            emptyTitle="Thư viện trống"
            emptyHint="Lưu truyện để quay lại nhanh hơn."
            action={
              <Link to="/library">
                <Button variant="ghost" size="sm">Mở thư viện</Button>
              </Link>
            }
          />
        </>
      )}
    </div>
  )
}


function WorkRail({
  title, icon: Icon, works, emptyTitle, emptyHint, action,
}: {
  title:      string
  icon:       LucideIcon
  works:      Array<{
    id: string
    title: string
    cover_url: string | null
    sources: { source: string }[]
    nsfw: boolean
  }>
  emptyTitle: string
  emptyHint:  string
  action?:    ReactNode
}) {
  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="inline-flex items-center gap-2 text-sm font-semibold text-text">
          <Icon size={17} className="text-text-subtle" />
          {title}
        </h2>
        {action}
      </div>

      {works.length === 0 ? (
        <div className="rounded-md bg-surface border border-border-soft">
          <EmptyState title={emptyTitle} hint={emptyHint} />
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 sm:gap-4">
          {works.map(work => (
            <WorkCard
              key={work.id}
              work={{
                id:        work.id,
                title:     work.title,
                cover_url: work.cover_url,
                source:    work.sources[0]?.source ?? null,
                nsfw:      work.nsfw,
              }}
            />
          ))}
        </div>
      )}
    </section>
  )
}


export const Route = createFileRoute('/')({
  component: HomePage,
  staticData: { auth: 'required' },
})
