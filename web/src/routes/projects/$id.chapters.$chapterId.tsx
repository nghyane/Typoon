import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { chaptersApi, chapterKeys } from '../../api/chapters'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { ProgressBar } from '../../components/ui/ProgressBar'
import type { ChapterOut } from '../../api/types'

export const Route = createFileRoute('/projects/$id/chapters/$chapterId')({
  component: ChapterPage,
})

function statusVariant(ch: ChapterOut): 'done' | 'running' | 'pending' | 'idle' | 'error' {
  if (ch.state === 'done')    return 'done'
  if (ch.state === 'error')   return 'error'
  if (ch.state === 'running') return 'running'
  if (ch.state === 'pending') return 'pending'
  return 'idle'
}

function ChapterPage() {
  const { id, chapterId } = Route.useParams()
  const projectId = Number(id)
  const chId      = Number(chapterId)
  const [pageIndex, setPageIndex] = useState(0)

  const { data: chapter } = useQuery({
    queryKey: chapterKeys.detail(projectId, chId),
    queryFn:  () => chaptersApi.get(projectId, chId),
    refetchInterval: (q) => q.state.data?.state === 'running' ? 2000 : false,
  })

  const isDone = chapter?.state === 'done'
  const pageUrl = chaptersApi.pageUrl(projectId, chId, pageIndex)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 h-12 border-b border-(--color-border) shrink-0">
        <Link
          to="/projects/$id"
          params={{ id }}
          className="flex items-center gap-1 text-xs text-(--color-text-3) hover:text-(--color-text-2) transition-colors"
        >
          <ChevronLeft size={13} />
          Danh sách chương
        </Link>
        <span className="text-(--color-border)">·</span>
        <span className="text-sm font-medium text-(--color-text-1)">
          {chapter ? `Chương ${chapter.idx}` : '—'}
        </span>
        {chapter && <StatusBadge variant={statusVariant(chapter)} />}
        {chapter?.state === 'running' && chapter.progress && (
          <div className="flex items-center gap-2 ml-1">
            <ProgressBar
              value={Math.round((chapter.progress.page_index / chapter.progress.page_total) * 100)}
              variant="running"
              className="w-24"
            />
            <span className="text-xs text-(--color-text-3)">
              {chapter.progress.page_index}/{chapter.progress.page_total}
            </span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto flex flex-col items-center py-8 px-4">
        {isDone ? (
          <>
            <img
              src={pageUrl}
              alt={`Page ${pageIndex + 1}`}
              className="max-w-2xl w-full rounded-xl shadow-sm border border-(--color-border)"
            />
            <div className="flex items-center gap-4 mt-4 px-4 py-2 rounded-xl border border-(--color-border) bg-(--color-bg)">
              <button
                disabled={pageIndex === 0}
                onClick={() => setPageIndex((i) => i - 1)}
                className="p-1 rounded-lg hover:bg-(--color-surface-1) transition-colors disabled:opacity-30 text-(--color-text-2)"
              >
                <ChevronLeft size={18} />
              </button>
              <span className="text-sm tabular-nums text-(--color-text-2)">
                {pageIndex + 1} / {chapter.page_count}
              </span>
              <button
                disabled={pageIndex >= chapter.page_count - 1}
                onClick={() => setPageIndex((i) => i + 1)}
                className="p-1 rounded-lg hover:bg-(--color-surface-1) transition-colors disabled:opacity-30 text-(--color-text-2)"
              >
                <ChevronRight size={18} />
              </button>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center justify-center flex-1 gap-2 text-(--color-text-3)">
            {chapter?.state === 'error'
              ? <span className="text-sm text-red-500">{chapter.error}</span>
              : <span className="text-sm">Chương chưa được render</span>}
          </div>
        )}
      </div>
    </div>
  )
}
