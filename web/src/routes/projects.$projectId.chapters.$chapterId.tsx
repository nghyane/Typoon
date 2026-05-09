import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { api, type ApiChapter } from '@shared/api/api'
import { Spinner } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { ReaderToolbar, type ViewMode } from '@features/chapter-reader/ReaderToolbar'
import { PageImage } from '@features/chapter-reader/PageImage'
import { useChapterArchive } from '@features/chapter-reader/useChapterArchive'
import { useProjectInterest } from '../store/interest'

interface SearchParams {
  page?: number
  mode?: ViewMode
}

function ChapterReaderPage() {
  const { projectId, chapterId } = Route.useParams()
  const { page = 0, mode = 'continuous' } = Route.useSearch()
  const nav = Route.useNavigate()
  const pid = Number(projectId)
  const cid = Number(chapterId)

  // Subscribe SSE to events for this project so render progress on the
  // chapter page reflects in real time.
  useProjectInterest(isNaN(pid) ? null : pid)

  const setPage = (p: number) =>
    nav({ search: (s) => ({ ...s, page: p > 0 ? p : undefined }) })
  const setMode = (m: ViewMode) =>
    nav({ search: (s) => ({ ...s, mode: m === 'continuous' ? undefined : m }) })

  // Project — for breadcrumb title.
  const { data: project } = useQuery({
    queryKey: ['projects', pid],
    queryFn:  () => api.getProject(pid),
    enabled:  !isNaN(pid),
  })

  // Chapter list — used to derive prev/next ids and current chapter meta.
  const { data: chapters = [] } = useQuery({
    queryKey: ['projects', pid, 'chapters'],
    queryFn:  () => api.listChapters(pid),
    enabled:  !isNaN(pid),
  })

  const sorted = useMemo(
    () => [...chapters].sort((a, b) => a.idx - b.idx),
    [chapters],
  )
  const current = sorted.find((c) => c.chapter_id === cid)
  const idxInList = current ? sorted.indexOf(current) : -1
  const prev: ApiChapter | null = idxInList > 0 ? sorted[idxInList - 1]! : null
  const next: ApiChapter | null = idxInList >= 0 && idxInList < sorted.length - 1 ? sorted[idxInList + 1]! : null

  // Archive — opens once per chapter. The URL embeds an updated_at
  // version so a re-render busts the CDN cache automatically.
  const { bunle, loading: aLoading, error: aError } = useChapterArchive(current?.archive_url)

  if (!current) {
    return (
      <div className="px-6 py-10 text-center">
        <p className="text-sm text-text-muted">Đang tải chương…</p>
      </div>
    )
  }

  if (current.state !== 'done') {
    return (
      <div className="px-6 py-16 text-center max-w-md mx-auto">
        <p className="text-sm font-medium text-text">Chương chưa render xong</p>
        <p className="text-xs text-text-subtle mt-1">
          Trạng thái hiện tại: {current.state}. Khi worker render xong, làm mới trang để xem.
        </p>
        <Link
          to="/projects/$projectId"
          params={{ projectId: String(pid) }}
          className="inline-block mt-4"
        >
          <Button>Quay về dự án</Button>
        </Link>
      </div>
    )
  }

  return (
    <div className="min-h-full bg-bg">
      <ReaderToolbar
        projectId={pid}
        projectTitle={project?.title ?? ''}
        chapterIdx={current.idx}
        chapterTitle={current.title}
        prevId={prev?.chapter_id ?? null}
        nextId={next?.chapter_id ?? null}
        page={page}
        totalPages={bunle?.pageCount ?? current.page_count}
        mode={mode}
        onModeChange={setMode}
      />

      {aLoading && (
        <div className="flex items-center justify-center py-24 text-text-subtle">
          <Spinner size={20} />
        </div>
      )}

      {aError && (
        <div className="py-16 text-center">
          <p className="text-sm text-error-text font-medium">Không tải được archive</p>
          <p className="text-xs text-text-subtle mt-1">{aError}</p>
        </div>
      )}

      {bunle && mode === 'continuous' && (
        <div className="max-w-3xl mx-auto py-4 space-y-1.5">
          {bunle.pages.map((info) => (
            <PageImage key={info.index} bunle={bunle} info={info} />
          ))}
        </div>
      )}

      {bunle && mode === 'single' && (
        <SinglePageView
          bunle={bunle}
          page={page}
          onChange={setPage}
        />
      )}
    </div>
  )
}

function SinglePageView({
  bunle, page, onChange,
}: {
  bunle:    NonNullable<ReturnType<typeof useChapterArchive>['bunle']>
  page:     number
  onChange: (p: number) => void
}) {
  const total = bunle.pageCount
  const safe  = Math.min(Math.max(0, page), total - 1)
  const info  = bunle.pages[safe]!

  return (
    <div className="max-w-3xl mx-auto py-4">
      <PageImage bunle={bunle} info={info} lazy={false} />
      <div className="flex items-center justify-between mt-4 px-2">
        <Button onClick={() => onChange(safe - 1)} disabled={safe <= 0}>
          ← Trang trước
        </Button>
        <span className="text-sm text-text-muted tabular">
          <span className="text-text font-medium">{safe + 1}</span>
          <span className="opacity-50 mx-1">/</span>
          {total}
        </span>
        <Button onClick={() => onChange(safe + 1)} disabled={safe >= total - 1}>
          Trang sau →
        </Button>
      </div>
    </div>
  )
}

export const Route = createFileRoute('/projects/$projectId/chapters/$chapterId')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    page: typeof s.page === 'number' && s.page > 0 ? s.page : undefined,
    mode: s.mode === 'single' ? 'single' : undefined,
  }),
  component: ChapterReaderPage,
})
