import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useQueries } from '@tanstack/react-query'
import { useState, useMemo } from 'react'
import { Plus, Search } from 'lucide-react'
import { api, type ApiChapter } from '../lib/api'
import { Cover } from '../components/Cover'

interface Stats { total: number; done: number; running: number; error: number }

function toStats(chs: ApiChapter[]): Stats {
  return chs.reduce(
    (acc, c) => {
      acc.total++
      if (c.state === 'done')    acc.done++
      if (c.state === 'running') acc.running++
      if (c.state === 'error')   acc.error++
      return acc
    },
    { total: 0, done: 0, running: 0, error: 0 },
  )
}

function pct(s: Stats) {
  return s.total === 0 ? 0 : Math.round(((s.done + s.error) / s.total) * 100)
}

function ProjectsPage() {
  const [q, setQ] = useState('')

  const { data: projects = [], isLoading, isError } = useQuery({
    queryKey: ['projects'],
    queryFn:  api.listProjects,
  })

  const chQ = useQueries({
    queries: projects.map((p) => ({
      queryKey: ['projects', p.project_id, 'chapters'],
      queryFn:  () => api.listChapters(p.project_id),
      enabled:  projects.length > 0,
    })),
  })

  const rows = useMemo(() => {
    const all = projects.map((p, i) => {
      const s = toStats(chQ[i]?.data ?? [])
      return { p, s, pct: pct(s) }
    })
    return all.filter(({ p }) =>
      p.title.toLowerCase().includes(q.toLowerCase()),
    )
  }, [projects, chQ, q])

  return (
    <div>

      {/* header */}
      <div className="px-6 pt-6 pb-5 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900">Dự án</h1>
          <p className="text-sm text-zinc-400 mt-1">Quản lý và theo dõi tiến độ dịch thuật</p>
        </div>
        <button className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 active:scale-95 transition-all cursor-pointer">
          <Plus size={13} />
          Thêm dự án
        </button>
      </div>

      {/* search */}
      <div className="px-6 mb-5">
        <label className="flex items-center gap-2 h-8 px-3 w-52 rounded-lg border border-zinc-200 hover:border-zinc-300 transition-colors cursor-text outline-none">
          <Search size={13} className="text-zinc-400 shrink-0" />
          <input
            type="text"
            placeholder="Tìm dự án..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-300"
          />
        </label>
      </div>

      {/* grid */}
      <div className="px-6">

        {/* skeleton */}
        {isLoading && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="w-full aspect-[2/3] rounded-xl bg-zinc-100 mb-2.5" />
                <div className="h-3 w-3/4 rounded bg-zinc-100 mb-1.5" />
                <div className="h-2.5 w-1/2 rounded bg-zinc-100" />
              </div>
            ))}
          </div>
        )}

        {/* error */}
        {isError && (
          <div className="py-10 text-center">
            <p className="text-sm text-red-500 font-medium">Không thể tải danh sách</p>
            <p className="text-xs text-zinc-400 mt-1">Kiểm tra kết nối rồi thử lại</p>
          </div>
        )}

        {/* cards */}
        {!isLoading && !isError && rows.length > 0 && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-4">
            {rows.map(({ p, s, pct: prog }) => (
              <Link
                key={p.project_id}
                to="/projects/$projectId"
                params={{ projectId: String(p.project_id) }}
                className="group"
              >
                {/* cover */}
                <Cover
                  src={p.cover_url}
                  title={p.title}
                  className="w-full aspect-[2/3] rounded-xl border border-zinc-200 mb-2.5 group-hover:border-zinc-300 transition-colors"
                />

                {/* meta */}
                <p className="text-sm font-medium text-zinc-900 truncate leading-snug">
                  {p.title}
                </p>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xs text-zinc-400">
                    {p.source_lang.toUpperCase()} → {p.target_lang.toUpperCase()}
                  </span>
                  <span className="text-xs text-zinc-400 tabular-nums">{s.total}</span>
                </div>

                {/* progress */}
                <div className="mt-2 h-1 rounded-full bg-zinc-100 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-zinc-300 transition-[width]"
                    style={{ width: `${prog}%` }}
                  />
                </div>
              </Link>
            ))}
          </div>
        )}

        {/* empty */}
        {!isLoading && !isError && rows.length === 0 && (
          <div className="py-16 flex flex-col items-center gap-3">
            <p className="text-sm font-medium text-zinc-500">
              {q ? 'Không tìm thấy kết quả' : 'Chưa có dự án nào'}
            </p>
            <p className="text-xs text-zinc-400">
              {q ? 'Thử từ khoá khác' : 'Tạo dự án để bắt đầu'}
            </p>
          </div>
        )}

      </div>
    </div>
  )
}

export const Route = createFileRoute('/projects/')({
  component: ProjectsPage,
})
