import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { RotateCcw, Eye, MoreHorizontal, Plus, ChevronLeft, Search } from 'lucide-react'
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

type Filter = 'all' | 'idle' | 'running' | 'done' | 'error'
type Tab = 'chapters' | 'pipeline'

function statusVariant(ch: ChapterOut): 'done' | 'running' | 'pending' | 'idle' | 'error' {
  if (ch.state === 'done')    return 'done'
  if (ch.state === 'error')   return 'error'
  if (ch.state === 'running') return 'running'
  if (ch.state === 'pending') return 'pending'
  return 'idle'
}

function chapterProgress(ch: ChapterOut): number {
  if (ch.state === 'done') return 100
  if (ch.progress) return Math.round((ch.progress.page_index / ch.progress.page_total) * 100)
  return 0
}

const STAGE_LABEL: Record<string, string> = {
  scan:      'Scanning',
  translate: 'Translating',
  render:    'Rendering',
}

function ProjectPage() {
  const { id } = Route.useParams()
  const projectId = Number(id)
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('chapters')
  const [filter, setFilter] = useState<Filter>('all')
  const [search, setSearch] = useState('')

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
    onError:   () => toast.error('Redo failed'),
  })

  const redoSelected = () => { selected.forEach((cid) => redo.mutate(cid)); clear() }

  const counts = {
    all:     chapters.length,
    idle:    chapters.filter((c) => c.state === 'idle').length,
    running: chapters.filter((c) => c.state === 'running' || c.state === 'pending').length,
    done:    chapters.filter((c) => c.state === 'done').length,
    error:   chapters.filter((c) => c.state === 'error').length,
  }

  const filtered = chapters
    .filter((c) => filter === 'all' ? true : filter === 'running'
      ? c.state === 'running' || c.state === 'pending'
      : c.state === filter)
    .filter((c) => search ? String(c.idx).includes(search) : true)

  const doneCount = counts.done
  const pct = chapters.length ? Math.round((doneCount / chapters.length) * 100) : 0

  const FILTERS: { key: Filter; label: string; count: number }[] = [
    { key: 'all',     label: 'All',          count: counts.all },
    { key: 'idle',    label: 'Not started',  count: counts.idle },
    { key: 'running', label: 'In progress',  count: counts.running },
    { key: 'done',    label: 'Done',         count: counts.done },
    { key: 'error',   label: 'Failed',       count: counts.error },
  ]

  const TABS: { key: Tab; label: string }[] = [
    { key: 'chapters', label: 'Chapters' },
    { key: 'pipeline', label: 'Pipeline' },
  ]

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-(--color-border) shrink-0">
        {/* Breadcrumb */}
        <div className="flex items-center gap-1.5 px-6 pt-4 pb-0 text-sm text-(--color-text-3)">
          <Link to="/projects" className="hover:text-(--color-accent) transition-colors flex items-center gap-1">
            <ChevronLeft size={13} />
            Projects
          </Link>
          <span>/</span>
          <span className="text-(--color-text-1) font-medium">{project?.title ?? '—'}</span>
        </div>

        {/* Project info */}
        <div className="flex items-start gap-4 px-6 pt-4 pb-4">
          <div className="w-12 h-12 rounded-lg shrink-0 flex items-center justify-center text-xl font-bold bg-(--color-surface) border border-(--color-border) text-(--color-text-2)">
            {project?.title?.[0] ?? '?'}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold text-(--color-text-1)">{project?.title ?? '—'}</h1>
              <span className="text-xs px-1.5 py-0.5 rounded border border-(--color-border) text-(--color-text-3)">
                {project?.source_lang?.toUpperCase()} → {project?.target_lang?.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1.5">
              <span className="text-sm text-(--color-text-3)">{chapters.length} chapters</span>
              <div className="flex items-center gap-2">
                <ProgressBar value={pct} variant="done" className="w-24" />
                <span className="text-xs text-(--color-text-3)">{pct}%</span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md border border-(--color-border) bg-(--color-bg) text-(--color-text-1) hover:bg-(--color-surface) transition-colors">
              <Plus size={14} />
              Add chapter
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex items-center gap-4 px-6">
          {TABS.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={cn(
                'pb-2.5 text-sm font-medium border-b-2 transition-colors',
                tab === t.key
                  ? 'border-(--color-text-1) text-(--color-text-1)'
                  : 'border-transparent text-(--color-text-3) hover:text-(--color-text-1)',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'pipeline' ? (
        <div className="flex-1 flex items-center justify-center text-(--color-text-3) text-sm">
          Pipeline view — coming soon
        </div>
      ) : (
        <>
          {/* Filters */}
          <div className="flex items-center justify-between px-6 py-2.5 border-b border-(--color-border) shrink-0">
            <div className="flex items-center gap-1">
              {FILTERS.map((f) => (
                <button
                  key={f.key}
                  onClick={() => setFilter(f.key)}
                  className={cn(
                    'flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-colors',
                    filter === f.key
                      ? 'bg-(--color-pill-active-bg) text-(--color-pill-active-text)'
                      : 'text-(--color-text-2) hover:bg-(--color-surface)',
                  )}
                >
                  {f.label}
                  {f.count > 0 && (
                    <span className={cn(
                      'text-[10px] rounded-full px-1.5 py-px',
                      filter === f.key ? 'bg-white/20 text-white' : 'bg-(--color-surface) text-(--color-text-3)',
                    )}>
                      {f.count}
                    </span>
                  )}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1.5 h-7 px-2.5 rounded-md border border-(--color-border) bg-(--color-bg)">
              <Search size={12} className="text-(--color-text-3) shrink-0" />
              <input
                placeholder="Search chapters..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="text-xs w-28 bg-transparent outline-none placeholder:text-(--color-text-3) text-(--color-text-1)"
              />
            </div>
          </div>

          {/* Table */}
          <div className="flex-1 overflow-y-auto">
            {/* Column header */}
            <div className="flex items-center gap-4 px-6 py-2 border-b border-(--color-border) sticky top-0 bg-(--color-bg) text-xs text-(--color-text-3)">
              <input type="checkbox" checked={isAllSelected} onChange={toggleAll} className="shrink-0" />
              <span className="w-12 shrink-0">#</span>
              <span className="flex-1">Chapter</span>
              <span className="w-28">Status</span>
              <span className="w-40">Progress</span>
              <span className="w-14 text-right">Pages</span>
              <span className="w-20 text-right">Actions</span>
            </div>

            {isLoading
              ? Array.from({ length: 10 }).map((_, i) => (
                  <div key={i} className="flex items-center gap-4 px-6 py-3 border-b border-(--color-border-subtle)">
                    <Skeleton className="w-4 h-4 shrink-0" />
                    <Skeleton className="w-12 h-3 shrink-0" />
                    <Skeleton className="flex-1 h-3" />
                    <Skeleton className="w-28 h-3" />
                    <Skeleton className="w-40 h-1.5" />
                    <Skeleton className="w-14 h-3" />
                    <Skeleton className="w-20 h-5" />
                  </div>
                ))
              : filtered.map((ch) => {
                  const variant = statusVariant(ch)
                  const pct     = chapterProgress(ch)
                  const isRunning = ch.state === 'running' || ch.state === 'pending'

                  return (
                    <div
                      key={ch.chapter_id}
                      className={cn(
                        'flex items-center gap-4 px-6 py-3 border-b border-(--color-border-subtle) transition-colors',
                        selected.has(ch.chapter_id) ? 'bg-blue-50' : 'hover:bg-(--color-surface)',
                      )}
                    >
                      <input
                        type="checkbox"
                        checked={selected.has(ch.chapter_id)}
                        onChange={() => toggle(ch.chapter_id)}
                        className="shrink-0"
                      />

                      <span className="w-12 text-sm text-(--color-text-3) shrink-0 tabular-nums">
                        {ch.idx}
                      </span>

                      <Link
                        to="/projects/$id/chapters/$chapterId"
                        params={{ id, chapterId: String(ch.chapter_id) }}
                        className="flex-1 text-sm text-(--color-accent) hover:underline truncate"
                      >
                        Chapter {ch.idx}
                      </Link>

                      <div className="w-28">
                        <StatusBadge
                          variant={variant}
                          label={isRunning && ch.stage ? STAGE_LABEL[ch.stage] ?? ch.stage : undefined}
                        />
                      </div>

                      <div className="w-40 flex items-center gap-2">
                        <ProgressBar value={pct} variant={variant} className="flex-1" />
                        <span className="text-xs w-7 text-right text-(--color-text-3) tabular-nums">{pct}%</span>
                      </div>

                      <span className="w-14 text-right text-sm text-(--color-text-2) tabular-nums">
                        {ch.page_count > 0 ? ch.page_count : '—'}
                      </span>

                      <div className="w-20 flex items-center justify-end gap-0.5">
                        <Link
                          to="/projects/$id/chapters/$chapterId"
                          params={{ id, chapterId: String(ch.chapter_id) }}
                          className="w-6 h-6 rounded flex items-center justify-center hover:bg-(--color-border-subtle) transition-colors text-(--color-text-3)"
                        >
                          <Eye size={13} />
                        </Link>
                        <button
                          onClick={() => redo.mutate(ch.chapter_id)}
                          className="w-6 h-6 rounded flex items-center justify-center hover:bg-(--color-border-subtle) transition-colors text-(--color-text-3)"
                        >
                          <RotateCcw size={13} />
                        </button>
                        <button className="w-6 h-6 rounded flex items-center justify-center hover:bg-(--color-border-subtle) transition-colors text-(--color-text-3)">
                          <MoreHorizontal size={13} />
                        </button>
                      </div>
                    </div>
                  )
                })}

            {!isLoading && (
              <p className="px-6 py-3 text-xs text-(--color-text-3)">
                Showing {filtered.length} of {chapters.length} chapters
              </p>
            )}
          </div>

          {/* Floating bulk bar */}
          {selected.size > 0 && (
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-3 px-4 py-2.5 rounded-xl border border-(--color-border) bg-(--color-bg) shadow-lg">
              <span className="text-sm text-(--color-text-2)">{selected.size} selected</span>
              <button
                onClick={redoSelected}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium bg-(--color-accent) text-white hover:bg-(--color-accent-hover) transition-colors"
              >
                <RotateCcw size={13} />
                Redo {selected.size} chapters
              </button>
              <button onClick={clear} className="text-sm text-(--color-text-3) hover:text-(--color-text-1) transition-colors">
                Cancel
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
