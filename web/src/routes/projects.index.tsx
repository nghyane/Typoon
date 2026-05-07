import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query'
import { useState, useMemo } from 'react'
import { Plus, Search, FolderOpen, Star, Globe, Users } from 'lucide-react'
import { api, type ProjectFilter, type ApiProject } from '../lib/api'
import { Cover } from '../components/Cover'
import { CreateProjectDialog } from '../components/CreateProjectDialog'
import { timeAgo } from '../lib/time'
import { cn } from '../lib/cn'

interface SearchParams { filter?: ProjectFilter }

const FILTER_LABEL: Record<ProjectFilter, string> = {
  mine:      'Của tôi',
  pinned:    'Đã lưu',
  community: 'Cộng đồng',
  all:       'Tất cả',
}

const EMPTY_HINT: Record<ProjectFilter, { title: string; sub: string }> = {
  mine:      { title: 'Chưa có dự án',             sub: 'Tạo dự án đầu tiên để bắt đầu' },
  pinned:    { title: 'Chưa lưu dự án nào',        sub: 'Bấm sao trên dự án để lưu xem sau' },
  community: { title: 'Chưa có dự án chia sẻ',     sub: 'Đợi thành viên khác chia sẻ dự án' },
  all:       { title: 'Chưa có dự án',             sub: 'Tạo dự án đầu tiên để bắt đầu' },
}

function ProjectsPage() {
  const { filter = 'mine' } = Route.useSearch()
  const [q,          setQ]          = useState('')
  const [createOpen, setCreateOpen] = useState(false)
  const qc = useQueryClient()

  const { data: projects = [], isLoading, isError } = useQuery({
    queryKey: ['projects', filter],
    queryFn:  () => api.listProjects(filter),
  })

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return projects
    return projects.filter((p) => p.title.toLowerCase().includes(needle))
  }, [projects, q])

  const pinMut = useMutation({
    mutationFn: ({ id, on }: { id: number; on: boolean }) =>
      on ? api.pinProject(id) : api.unpinProject(id),
    onMutate: async ({ id, on }) => {
      // Optimistic toggle across all filter caches.
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
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900">
            {FILTER_LABEL[filter]}
          </h1>
          <p className="text-sm text-zinc-400 mt-1">
            <FilterHint filter={filter} count={projects.length} />
          </p>
        </div>
        {filter === 'mine' && (
          <button
            onClick={() => setCreateOpen(true)}
            className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 active:scale-[0.98] transition-all cursor-pointer"
          >
            <Plus size={13} />
            Thêm dự án
          </button>
        )}
      </div>

      {/* search */}
      <div className="px-6 mb-5">
        <label className="flex items-center gap-2 h-9 px-3 max-w-sm rounded-lg border border-zinc-200 hover:border-zinc-300 focus-within:border-zinc-400 transition-colors cursor-text">
          <Search size={14} className="text-zinc-400 shrink-0" />
          <input
            type="text"
            placeholder="Tìm dự án..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-300"
          />
          {q && (
            <button
              onClick={() => setQ('')}
              className="text-xs text-zinc-400 hover:text-zinc-600 cursor-pointer"
            >
              Xoá
            </button>
          )}
        </label>
      </div>

      {/* grid */}
      <div className="px-6 pb-8">
        {isLoading && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(168px,1fr))] gap-x-4 gap-y-6">
            {Array.from({ length: 12 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="w-full aspect-[2/3] rounded-xl bg-zinc-100 mb-2.5" />
                <div className="h-3 w-3/4 rounded bg-zinc-100 mb-1.5" />
                <div className="h-2.5 w-1/2 rounded bg-zinc-100" />
              </div>
            ))}
          </div>
        )}

        {isError && (
          <div className="py-16 text-center">
            <p className="text-sm text-red-500 font-medium">Không thể tải danh sách</p>
            <p className="text-xs text-zinc-400 mt-1">Kiểm tra kết nối rồi thử lại</p>
          </div>
        )}

        {!isLoading && !isError && filtered.length > 0 && (
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

        {!isLoading && !isError && filtered.length === 0 && (
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
          className="w-full aspect-[2/3] rounded-xl border border-zinc-200 mb-3 group-hover:border-zinc-300 group-hover:shadow-sm transition-all"
        />
        <p className="text-sm font-medium text-zinc-900 leading-snug line-clamp-2 group-hover:text-zinc-700 transition-colors">
          {p.title}
        </p>
        <div className="flex items-center justify-between mt-1.5 text-xs text-zinc-400">
          <span className="uppercase tracking-wide flex items-center gap-1.5">
            {p.source_lang} → {p.target_lang}
            {p.shared && (
              <Globe size={10} className="text-emerald-500" aria-label="Đã chia sẻ" />
            )}
          </span>
          {p.updated_at && (
            <span className="tabular-nums" title={p.updated_at}>
              {timeAgo(p.updated_at)}
            </span>
          )}
        </div>
      </Link>

      {/* Pin star — only on shared+non-owned (community) and shared own (so user can unpin from grid). */}
      {(p.shared || p.is_pinned) && (
        <button
          onClick={(e) => {
            e.preventDefault()
            e.stopPropagation()
            onTogglePin(!p.is_pinned)
          }}
          title={p.is_pinned ? 'Bỏ lưu' : 'Lưu'}
          className={cn(
            'absolute top-2 right-2 size-7 rounded-lg flex items-center justify-center cursor-pointer transition-all',
            'bg-white/80 backdrop-blur border border-zinc-200/80',
            'opacity-0 group-hover:opacity-100',
            p.is_pinned && 'opacity-100',
          )}
        >
          <Star
            size={13}
            className={cn(
              p.is_pinned ? 'fill-amber-400 text-amber-400' : 'text-zinc-500',
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
      <div className="size-12 rounded-2xl bg-zinc-100 flex items-center justify-center mb-3">
        <Icon size={20} className="text-zinc-400" />
      </div>
      <p className="text-sm font-medium text-zinc-700">{hint.title}</p>
      <p className="text-xs text-zinc-400 mt-1">{hint.sub}</p>
      {!query && filter === 'mine' && (
        <button
          onClick={onAdd}
          className="mt-4 inline-flex items-center gap-1.5 h-8 px-4 rounded-lg bg-zinc-900 text-white text-xs font-medium hover:bg-zinc-700 cursor-pointer"
        >
          <Plus size={12} />
          Thêm dự án
        </button>
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
