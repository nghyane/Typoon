import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState, useMemo } from 'react'
import { Plus, Search, FolderOpen } from 'lucide-react'
import { api } from '../lib/api'
import { Cover } from '../components/Cover'
import { CreateProjectDialog } from '../components/CreateProjectDialog'
import { timeAgo } from '../lib/time'

function ProjectsPage() {
  const [q,        setQ]        = useState('')
  const [createOpen, setCreateOpen] = useState(false)

  const { data: projects = [], isLoading, isError } = useQuery({
    queryKey: ['projects'],
    queryFn:  api.listProjects,
  })

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    if (!needle) return projects
    return projects.filter((p) => p.title.toLowerCase().includes(needle))
  }, [projects, q])

  return (
    <div>
      {/* header */}
      <div className="px-6 pt-6 pb-5 flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900">Dự án</h1>
          <p className="text-sm text-zinc-400 mt-1">
            {projects.length > 0
              ? `${projects.length} dự án • cập nhật theo thời gian thực`
              : 'Quản lý và theo dõi tiến độ dịch thuật'}
          </p>
        </div>
        <button
          onClick={() => setCreateOpen(true)}
          className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 active:scale-[0.98] transition-all cursor-pointer"
        >
          <Plus size={13} />
          Thêm dự án
        </button>
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
              <Link
                key={p.project_id}
                to="/projects/$projectId"
                params={{ projectId: String(p.project_id) }}
                className="group block"
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
                  <span className="uppercase tracking-wide">
                    {p.source_lang} → {p.target_lang}
                  </span>
                  {p.updated_at && (
                    <span className="tabular-nums" title={p.updated_at}>
                      {timeAgo(p.updated_at)}
                    </span>
                  )}
                </div>
              </Link>
            ))}
          </div>
        )}

        {!isLoading && !isError && filtered.length === 0 && (
          <EmptyState query={q} onAdd={() => setCreateOpen(true)} />
        )}
      </div>

      <CreateProjectDialog open={createOpen} onClose={() => setCreateOpen(false)} />
    </div>
  )
}

function EmptyState({ query, onAdd }: { query: string; onAdd: () => void }) {
  return (
    <div className="py-20 flex flex-col items-center text-center">
      <div className="size-12 rounded-2xl bg-zinc-100 flex items-center justify-center mb-3">
        <FolderOpen size={20} className="text-zinc-400" />
      </div>
      <p className="text-sm font-medium text-zinc-700">
        {query ? 'Không tìm thấy kết quả' : 'Chưa có dự án nào'}
      </p>
      <p className="text-xs text-zinc-400 mt-1">
        {query ? 'Thử từ khoá khác' : 'Tạo dự án đầu tiên để bắt đầu dịch'}
      </p>
      {!query && (
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
  component: ProjectsPage,
})
