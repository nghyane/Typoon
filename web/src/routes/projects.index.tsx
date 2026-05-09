import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import { useState, useMemo } from 'react'
import { Plus, Search, FolderOpen, Star, Globe, Users } from 'lucide-react'
import { api, type ProjectFilter, type ApiProject } from '@shared/api/api'
import { Cover } from '@shared/ui/Cover'
import { Button } from '@shared/ui/Button'
import { input as inputCls } from '@shared/ui/primitives'
import { useDelayedFlag } from '@shared/lib/useDelayedFlag'
import { CreateProjectDialog } from '@features/projects-list/CreateProjectDialog'
import { timeAgo } from '@shared/lib/time'
import { cn } from '@shared/lib/cn'

interface SearchParams { filter?: ProjectFilter }

const FILTER_LABEL: Record<ProjectFilter, string> = {
  mine:      'Của tôi',
  pinned:    'Đã lưu',
  community: 'Cộng đồng',
  all:       'Tất cả',
}

const EMPTY_HINT: Record<ProjectFilter, { title: string; sub: string }> = {
  mine:      { title: 'Chưa có dự án',         sub: 'Tạo dự án đầu tiên để bắt đầu' },
  pinned:    { title: 'Chưa lưu dự án nào',    sub: 'Bấm sao trên dự án để lưu xem sau' },
  community: { title: 'Chưa có dự án chia sẻ', sub: 'Đợi thành viên khác chia sẻ dự án' },
  all:       { title: 'Chưa có dự án',         sub: 'Tạo dự án đầu tiên để bắt đầu' },
}

function ProjectsPage() {
  const { filter = 'mine' } = Route.useSearch()
  const [q,          setQ]          = useState('')
  const [createOpen, setCreateOpen] = useState(false)
  const qc = useQueryClient()

  const { data: projects = [], isPending: isLoading, isError } = useQuery({
    queryKey: ['projects', filter],
    queryFn:  () => api.listProjects(filter),
    // SSE is project-scoped and only opens for the project the user
    // is actually viewing. The list page falls back to polling so
    // chapter counts and state badges stay current without a
    // shotgun subscription. RQ pauses polling when the tab is in the
    // background (refetchOnWindowFocus default).
    refetchInterval: 10_000,
  })

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return projects
    return projects.filter((p) => p.title.toLowerCase().includes(needle))
  }, [projects, q])

  const showSkeleton = useDelayedFlag(isLoading, 250)
  const showEmpty    = !isLoading && !isError && filtered.length === 0
  const showGrid     = !isLoading && !isError && filtered.length > 0

  const pinMut = useMutation({
    mutationFn: ({ id, on }: { id: number; on: boolean }) =>
      on ? api.pinProject(id) : api.unpinProject(id),
    onMutate: async ({ id, on }) => {
      await qc.cancelQueries({ queryKey: ['projects'] })
      qc.setQueriesData<ApiProject[]>({ queryKey: ['projects'] }, (rows) =>
        rows?.map((p) => (p.project_id === id ? { ...p, is_pinned: on } : p))
      )
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['projects'] })
    },
  })

  return (
    <div>
      {/* header */}
      <div className="px-6 pt-6 pb-5 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">
            {FILTER_LABEL[filter]}
          </h1>
          <p className="text-sm text-text-subtle mt-1">
            <FilterHint filter={filter} count={projects.length} />
          </p>
        </div>
        {filter === 'mine' && (
          <Button variant="primary" onClick={() => setCreateOpen(true)}>
            <Plus size={13} />
            Thêm dự án
          </Button>
        )}
      </div>

      {/* search */}
      <div className="px-6 mb-5">
        <label className={cn(inputCls, 'flex items-center gap-2 max-w-sm cursor-text')}>
          <Search size={14} className="text-text-subtle shrink-0" />
          <input
            type="text"
            placeholder="Tìm dự án…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-text-subtle text-text min-w-0"
          />
          {q && (
            <button
              onClick={() => setQ('')}
              className="text-xs text-text-subtle hover:text-text cursor-pointer"
            >
              Xoá
            </button>
          )}
        </label>
      </div>

      {/* grid */}
      <div className="px-6 pb-8">
        {showSkeleton && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-x-4 gap-y-6">
            {Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="w-full aspect-[2/3] rounded-md bg-surface-2 mb-2.5" />
                <div className="h-3 w-3/4 rounded bg-surface-2 mb-1.5" />
                <div className="h-2.5 w-1/2 rounded bg-surface-2" />
              </div>
            ))}
          </div>
        )}

        {isError && (
          <div className="py-16 text-center">
            <p className="text-sm text-error-text font-medium">Không thể tải danh sách</p>
            <p className="text-xs text-text-subtle mt-1">Kiểm tra kết nối rồi thử lại</p>
          </div>
        )}

        {showGrid && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-x-4 gap-y-6">
            {filtered.map((p) => (
              <ProjectCard
                key={p.project_id}
                project={p}
                onTogglePin={(on) => pinMut.mutate({ id: p.project_id, on })}
              />
            ))}
          </div>
        )}

        {showEmpty && (
          <EmptyState filter={filter} query={q} onAdd={() => setCreateOpen(true)} />
        )}
      </div>

      <CreateProjectDialog open={createOpen} onClose={() => setCreateOpen(false)} />
    </div>
  )
}

function FilterHint({ filter, count }: { filter: ProjectFilter; count: number }) {
  if (count === 0) {
    if (filter === 'community') return <>Khám phá dự án thành viên khác đã chia sẻ</>
    if (filter === 'pinned')    return <>Lưu dự án để theo dõi tiến độ</>
    return <>Quản lý dự án dịch của bạn</>
  }
  return <>{count} dự án</>
}

function ProjectCard({
  project: p, onTogglePin,
}: {
  project: ApiProject
  onTogglePin: (on: boolean) => void
}) {
  return (
    <div className="group relative">
      <Link
        to="/projects/$projectId"
        params={{ projectId: String(p.project_id) }}
        className="block"
      >
        <Cover
          src={p.cover_url}
          title={p.title}
          version={p.updated_at}
          className="w-full aspect-[2/3] rounded-md mb-3 group-hover:brightness-110 transition-[filter]"
        />
        <p className="text-sm font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
          {p.title}
        </p>
        <div className="flex items-center justify-between mt-1.5 text-xs text-text-subtle">
          <span className="uppercase tracking-wide flex items-center gap-1.5">
            {p.source_lang} → {p.target_lang}
            {p.shared && (
              <Globe size={10} className="text-success-text" aria-label="Đã chia sẻ" />
            )}
          </span>
          {p.updated_at && (
            <span className="tabular" title={p.updated_at}>
              {timeAgo(p.updated_at)}
            </span>
          )}
        </div>
      </Link>

      {/* Pin star — only on shared+non-owned and shared own. */}
      {(p.shared || p.is_pinned) && (
        <button
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            onTogglePin(!p.is_pinned)
          }}
          title={p.is_pinned ? 'Bỏ lưu' : 'Lưu'}
          className={cn(
            'absolute top-2 right-2 size-7 rounded-sm flex items-center justify-center cursor-pointer',
            'bg-bg/80 backdrop-blur transition-opacity',
            'opacity-0 group-hover:opacity-100',
            p.is_pinned && 'opacity-100',
          )}
        >
          <Star
            size={13}
            className={cn(
              p.is_pinned ? 'fill-warning text-warning' : 'text-text-muted',
            )}
          />
        </button>
      )}
    </div>
  )
}

function EmptyState({
  filter, query, onAdd,
}: {
  filter: ProjectFilter
  query: string
  onAdd: () => void
}) {
  const Icon = filter === 'pinned'    ? Star
             : filter === 'community' ? Users
             : FolderOpen
  const hint = query
    ? { title: 'Không tìm thấy kết quả', sub: 'Thử từ khoá khác' }
    : EMPTY_HINT[filter]

  return (
    <div className="py-20 flex flex-col items-center text-center">
      <div className="size-12 rounded-md bg-surface flex items-center justify-center mb-3">
        <Icon size={20} className="text-text-subtle" />
      </div>
      <p className="text-sm font-medium text-text">{hint.title}</p>
      <p className="text-xs text-text-subtle mt-1">{hint.sub}</p>
      {!query && filter === 'mine' && (
        <Button variant="primary" onClick={onAdd} className="mt-4">
          <Plus size={12} />
          Thêm dự án
        </Button>
      )}
    </div>
  )
}

export const Route = createFileRoute('/projects/')({
  validateSearch: (search: Record<string, unknown>): SearchParams => {
    const f = search.filter
    return {
      filter: (f === 'mine' || f === 'pinned' || f === 'community' || f === 'all')
        ? f
        : 'mine',
    }
  },
  component: ProjectsPage,
})
