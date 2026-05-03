import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { RotateCcw, Eye, MoreHorizontal, Share2, Plus, ChevronLeft, SlidersHorizontal, Search } from 'lucide-react'
import { projectsApi, projectKeys } from '../../api/projects'
import { chaptersApi, chapterKeys } from '../../api/chapters'
import { useBulkSelect } from '../../hooks/useBulkSelect'
import { toast } from '../../stores/toast'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { ProgressBar } from '../../components/ui/ProgressBar'
import { Skeleton } from '../../components/ui/Skeleton'
import { cn } from '../../lib/cn'
import type { ChapterOut } from '../../api/types'

export const Route = createFileRoute('/projects/$id')({
  component: ProjectPage,
})

type Filter = 'all' | 'idle' | 'running' | 'done'
type TabKey = 'chapters' | 'pipeline' | 'library'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'chapters', label: 'Tất cả chương' },
  { key: 'pipeline', label: 'Pipeline' },
  { key: 'library',  label: 'Thư viện' },
]

function statusVariant(ch: ChapterOut): 'done' | 'running' | 'pending' | 'idle' | 'error' {
  if (ch.state === 'done')    return 'done'
  if (ch.state === 'error')   return 'error'
  if (ch.state === 'running') return 'running'
  if (ch.state === 'pending') return 'pending'
  return 'idle'
}

function chapterProgress(ch: ChapterOut): number {
  if (ch.state === 'done') return 100
  if (ch.state === 'idle') return 0
  if (ch.progress) return Math.round((ch.progress.page_index / ch.progress.page_total) * 100)
  return 0
}

function ProjectPage() {
  const { id } = Route.useParams()
  const projectId = Number(id)
  const qc = useQueryClient()
  const [tab, setTab] = useState<TabKey>('chapters')
  const [filter, setFilter] = useState<Filter>('all')

  const { data: project } = useQuery({
    queryKey: projectKeys.detail(projectId),
    queryFn:  () => projectsApi.get(projectId),
  })

  const { data: chapters = [], isLoading } = useQuery({
    queryKey: chapterKeys.all(projectId),
    queryFn:  () => chaptersApi.list(projectId),
    refetchInterval: 5000,
  })

  const { selected, toggle, toggleAll, clear, isAllSelected } = useBulkSelect(chapters)

  const redo = useMutation({
    mutationFn: (chapterId: number) => chaptersApi.redo(projectId, chapterId),
    onSuccess: () => qc.invalidateQueries({ queryKey: chapterKeys.all(projectId) }),
    onError:   () => toast.error('Redo thất bại'),
  })

  const redoSelected = () => {
    selected.forEach((cid) => redo.mutate(cid))
    clear()
  }

  const counts = {
    all:     chapters.length,
    idle:    chapters.filter((c) => c.state === 'idle').length,
    running: chapters.filter((c) => c.state === 'running' || c.state === 'pending').length,
    done:    chapters.filter((c) => c.state === 'done').length,
  }

  const filtered = filter === 'all'
    ? chapters
    : filter === 'running'
      ? chapters.filter((c) => c.state === 'running' || c.state === 'pending')
      : chapters.filter((c) => c.state === filter)

  const doneCount = counts.done
  const total = chapters.length
  const pct = total ? Math.round((doneCount / total) * 100) : 0

  const FILTERS: { key: Filter; label: string }[] = [
    { key: 'all',     label: `All` },
    { key: 'idle',    label: `Chưa dịch` },
    { key: 'running', label: `Đang xử lý` },
    { key: 'done',    label: `Hoàn thành` },
  ]

  return (
    <div className="flex flex-col h-full">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 px-6 h-10 border-b border-(--color-border) shrink-0">
        <Link to="/projects" className="flex items-center gap-1 text-xs text-(--color-text-3) hover:text-(--color-text-2) transition-colors">
          <ChevronLeft size={13} />
          Dự án
        </Link>
      </div>

      {/* Project header */}
      <div className="px-6 pt-5 pb-0 border-b border-(--color-border) shrink-0">
        <div className="flex items-start gap-4 mb-4">
          {/* Cover */}
          <div className="w-20 h-20 rounded-xl shrink-0 flex items-center justify-center text-3xl font-bold bg-(--color-surface-2) text-(--color-text-3)">
            {project?.title?.[0] ?? '?'}
          </div>

          <div className="flex-1 min-w-0 pt-1">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <h1 className="text-xl font-bold text-(--color-text-1)">{project?.title ?? '—'}</h1>
            </div>

            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm px-2 py-0.5 rounded-md font-medium bg-(--color-surface-2) text-(--color-text-2)">
                {project?.source_lang?.toUpperCase()}
              </span>
              <span className="text-(--color-text-3) text-sm">→</span>
              <span className="text-sm px-2 py-0.5 rounded-md font-medium bg-(--color-surface-2) text-(--color-text-2)">
                {project?.target_lang?.toUpperCase()}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs text-(--color-text-3)">{total} chương</span>
              <div className="flex items-center gap-2 flex-1 max-w-48">
                <ProgressBar value={pct} variant="done" className="flex-1" />
                <span className="text-xs font-medium text-(--color-text-2)">{pct}%</span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 shrink-0 pt-1">
            <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-(--color-border) bg-(--color-bg) text-(--color-text-1) hover:bg-(--color-surface-1) transition-colors">
              <Share2 size={13} />
              Chia sẻ
            </button>
            <button className="w-7 h-7 rounded-lg flex items-center justify-center border border-(--color-border) hover:bg-(--color-surface-1) transition-colors text-(--color-text-2)">
              <MoreHorizontal size={14} />
            </button>
            <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-(--color-accent) text-(--color-accent-text) hover:bg-(--color-accent-hover) transition-colors">
              <Plus size={13} />
              Thêm chương
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-1">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={cn(
                'px-4 py-2.5 text-sm font-medium border-b-2 transition-colors',
                tab === t.key
                  ? 'border-(--color-text-1) text-(--color-text-1)'
                  : 'border-transparent text-(--color-text-3) hover:text-(--color-text-2)',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Filter + search row */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-(--color-border) shrink-0">
        <div className="flex items-center gap-1.5">
          {FILTERS.map((f) => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={cn(
                'px-3 py-1 rounded-full text-xs font-medium transition-colors',
                filter === f.key
                  ? 'bg-(--color-pill-active-bg) text-(--color-pill-active-text)'
                  : 'text-(--color-text-2) hover:bg-(--color-surface-1)',
              )}
            >
              {f.label}
              {f.key !== 'all' && (
                <span className="ml-1 text-(--color-text-3)">
                  {f.key === 'idle' ? counts.idle : f.key === 'running' ? counts.running : counts.done}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 h-8 rounded-lg border border-(--color-border) bg-(--color-bg)">
            <Search size={13} className="text-(--color-text-3)" />
            <input
              placeholder="Tìm chương..."
              className="text-xs bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text-1) w-32"
            />
          </div>
          <button className="w-8 h-8 rounded-lg flex items-center justify-center border border-(--color-border) hover:bg-(--color-surface-1) transition-colors text-(--color-text-2)">
            <SlidersHorizontal size={14} />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        {/* Column headers */}
        <div className="flex items-center gap-4 px-6 py-2 border-b border-(--color-border) sticky top-0 bg-(--color-bg)">
          <input type="checkbox" checked={isAllSelected} onChange={toggleAll} className="shrink-0" />
          <span className="flex-1 text-xs text-(--color-text-3)">Chương</span>
          <span className="w-32 text-xs text-(--color-text-3)">Trạng thái</span>
          <span className="w-44 text-xs text-(--color-text-3)">Tiến độ</span>
          <span className="w-16 text-xs text-right text-(--color-text-3)">Số trang</span>
          <span className="w-28 text-xs text-right text-(--color-text-3)">Cập nhật lần cuối</span>
          <span className="w-20 text-xs text-right text-(--color-text-3)">Thao tác</span>
        </div>

        {isLoading
          ? Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4 px-6 py-3.5 border-b border-(--color-border-subtle)">
                <Skeleton className="w-4 h-4 rounded" />
                <Skeleton className="flex-1 h-4 rounded" />
                <Skeleton className="w-32 h-4 rounded" />
                <Skeleton className="w-44 h-2 rounded" />
                <Skeleton className="w-16 h-4 rounded" />
                <Skeleton className="w-28 h-4 rounded" />
                <Skeleton className="w-20 h-4 rounded" />
              </div>
            ))
          : filtered.map((ch) => {
              const variant = statusVariant(ch)
              const pct = chapterProgress(ch)
              const isSelected = selected.has(ch.chapter_id)

              return (
                <div
                  key={ch.chapter_id}
                  className={cn(
                    'flex items-center gap-4 px-6 py-3.5 border-b border-(--color-border-subtle) transition-colors',
                    isSelected ? 'bg-blue-50' : 'hover:bg-(--color-surface-1)',
                  )}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggle(ch.chapter_id)}
                    className="shrink-0"
                  />

                  <div className="flex-1 flex items-center gap-3 min-w-0">
                    <span className="text-sm text-(--color-text-3) w-10 shrink-0">{ch.idx}</span>
                    <Link
                      to="/projects/$id/chapters/$chapterId"
                      params={{ id, chapterId: String(ch.chapter_id) }}
                      className="text-sm text-(--color-text-1) hover:underline truncate"
                    >
                      Chương {ch.idx}
                    </Link>
                  </div>

                  <div className="w-32">
                    <StatusBadge variant={variant} />
                  </div>

                  <div className="w-44 flex items-center gap-2">
                    <ProgressBar value={pct} variant={variant} className="flex-1" />
                    <span className="text-xs w-9 text-right shrink-0 text-(--color-text-3)">{pct}%</span>
                  </div>

                  <span className="w-16 text-right text-sm text-(--color-text-2)">
                    {ch.page_count > 0 ? ch.page_count : '—'}
                  </span>

                  <span className="w-28 text-right text-sm text-(--color-text-3)">—</span>

                  <div className="w-20 flex items-center justify-end gap-1">
                    <Link
                      to="/projects/$id/chapters/$chapterId"
                      params={{ id, chapterId: String(ch.chapter_id) }}
                      className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-(--color-surface-2) transition-colors text-(--color-text-3)"
                    >
                      <Eye size={14} />
                    </Link>
                    <button
                      onClick={() => redo.mutate(ch.chapter_id)}
                      className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-(--color-surface-2) transition-colors text-(--color-text-3)"
                    >
                      <RotateCcw size={14} />
                    </button>
                    <button className="w-7 h-7 rounded-lg flex items-center justify-center hover:bg-(--color-surface-2) transition-colors text-(--color-text-3)">
                      <MoreHorizontal size={14} />
                    </button>
                  </div>
                </div>
              )
            })}

        {/* Footer count */}
        {!isLoading && (
          <div className="px-6 py-3 text-xs text-(--color-text-3)">
            Hiển thị 1–{filtered.length} trong tổng số {total} chương
          </div>
        )}
      </div>

      {/* Floating bulk bar */}
      {selected.size > 0 && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 px-5 py-3 rounded-2xl shadow-xl border border-(--color-border) bg-(--color-bg)">
          <span className="text-sm text-(--color-text-2)">{selected.size} chương đã chọn</span>
          <button
            onClick={redoSelected}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold bg-(--color-accent) text-(--color-accent-text) hover:bg-(--color-accent-hover) transition-colors"
          >
            <RotateCcw size={14} />
            Dịch {selected.size} chương đã chọn
          </button>
          <button
            onClick={clear}
            className="w-7 h-7 rounded-full flex items-center justify-center hover:bg-(--color-surface-2) transition-colors text-(--color-text-3)"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  )
}
