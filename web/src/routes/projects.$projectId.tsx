import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery } from '@tanstack/react-query'
import { useState, useMemo, useEffect } from 'react'
import { useHeaderStore } from '../store/header'
import {
  Plus, Search, Share2, MoreHorizontal, Download,
  Check, Eye, Pencil, ChevronDown,
  BookOpen, Clock, SlidersHorizontal,
  Type, Layers, Palette, FileText, Settings2,
} from 'lucide-react'
import { api } from '../lib/api'
import { cn } from '../lib/cn'
import { timeAgo } from '../lib/time'
import { Cover } from '../components/Cover'

// ── types ─────────────────────────────────────────────────────────────────────

type Tab = 'chapters' | 'context'
type Filter = 'all' | 'idle' | 'running' | 'done'

// ── helpers ───────────────────────────────────────────────────────────────────

function statusOf(state: string) {
  switch (state) {
    case 'done':    return { label: 'Hoàn thành', dot: 'bg-green-500', text: 'text-zinc-700', bar: 'bg-green-500' }
    case 'running': return { label: 'Đang dịch',  dot: 'bg-blue-500',  text: 'text-zinc-700', bar: 'bg-blue-500'  }
    case 'error':   return { label: 'Lỗi',        dot: 'bg-red-500',   text: 'text-zinc-700', bar: 'bg-red-500'   }
    default:        return { label: 'Chờ xử lý',  dot: 'bg-zinc-300',  text: 'text-zinc-400', bar: 'bg-zinc-300'  }
  }
}

// ── button variants ───────────────────────────────────────────────────────────

const btn = {
  primary:   'inline-flex items-center gap-1.5 h-9 px-4 rounded-lg bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 active:scale-95 transition-all cursor-pointer',
  secondary: 'inline-flex items-center gap-1.5 h-9 px-3.5 rounded-lg border border-zinc-200 text-sm text-zinc-600 hover:bg-zinc-100 hover:border-zinc-300 transition-colors cursor-pointer',
  icon:      'size-9 rounded-lg flex items-center justify-center border border-zinc-200 text-zinc-500 hover:bg-zinc-100 hover:border-zinc-300 transition-colors cursor-pointer',
  ghost:     'size-7 rounded-md flex items-center justify-center text-zinc-300 hover:text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer',
} as const

// ── static config ─────────────────────────────────────────────────────────────

const TABS: { key: Tab; label: string }[] = [
  { key: 'chapters', label: 'Chương'   },
  { key: 'context',  label: 'Ngữ cảnh' },
]

const COLS: { label: string; className?: string }[] = [
  { label: 'Chương'             },
  { label: 'Trang',             className: 'w-20' },
  { label: 'Trạng thái',        className: 'w-32' },
  { label: 'Cập nhật lần cuối', className: 'w-32' },
  { label: 'Thao tác',          className: 'w-20' },
]

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all',     label: 'All'        },
  { key: 'idle',    label: 'Chờ xử lý'  },
  { key: 'running', label: 'Đang dịch'  },
  { key: 'done',    label: 'Hoàn thành' },
]

// ── tab content ───────────────────────────────────────────────────────────────

function PipelineTab() {
  const items = [
    { icon: Type,      label: 'Thuật ngữ', desc: 'Từ vựng dịch chuyên ngành' },
    { icon: Layers,    label: 'Nhân vật',  desc: 'Tên, tính cách, xưng hô'   },
    { icon: Palette,   label: 'Font chữ',  desc: 'Font cấu hình cho dự án'   },
    { icon: FileText,  label: 'Prompts',   desc: 'System prompts cho LLM'     },
    { icon: Settings2, label: 'Cấu hình',  desc: 'Pipeline settings'          },
  ]
  return (
    <div className="grid grid-cols-3 gap-3">
      {items.map(({ icon: Icon, label, desc }) => (
        <button
          key={label}
          className="flex items-start gap-3 p-4 rounded-xl border border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 text-left cursor-pointer transition-colors"
        >
          <span className="size-8 rounded-lg bg-zinc-100 flex items-center justify-center shrink-0">
            <Icon size={14} className="text-zinc-500" />
          </span>
          <div>
            <p className="text-sm font-medium text-zinc-900">{label}</p>
            <p className="text-xs text-zinc-400 mt-0.5">{desc}</p>
          </div>
        </button>
      ))}
    </div>
  )
}

// ── page ──────────────────────────────────────────────────────────────────────

function ProjectDetailPage() {
  const { projectId } = Route.useParams()
  const id = Number(projectId)

  const [tab, setTab]       = useState<Tab>('chapters')
  const [filter, setFilter] = useState<Filter>('all')
  const [q, setQ]           = useState('')
  const [sel, setSel]       = useState<Set<number>>(new Set())
  const setHeader           = useHeaderStore((s) => s.set)
  const clearHeader         = useHeaderStore((s) => s.clear)

  const { data: project, isLoading: pLoad, isError: pErr } = useQuery({
    queryKey: ['projects', id],
    queryFn:  () => api.getProject(id),
    enabled:  !isNaN(id),
  })

  const { data: chapters = [], isLoading: cLoad } = useQuery({
    queryKey: ['projects', id, 'chapters'],
    queryFn:  () => api.listChapters(id),
    enabled:  !isNaN(id),
  })

  const filtered = useMemo(() => {
    let list = chapters
    if (filter !== 'all') {
      list = list.filter((c) =>
        filter === 'idle'
          ? c.state === 'idle' || c.state === 'pending'
          : c.state === filter,
      )
    }
    if (q.trim()) list = list.filter((c) => String(c.idx).includes(q))
    return list
  }, [chapters, filter, q])

  const toggleOne = (cid: number) =>
    setSel((prev) => {
      const next = new Set(prev)
      next.has(cid) ? next.delete(cid) : next.add(cid)
      return next
    })

  const toggleAll = () =>
    sel.size === filtered.length && filtered.length > 0
      ? setSel(new Set())
      : setSel(new Set(filtered.map((c) => c.chapter_id)))

  const total      = chapters.length
  const done       = chapters.filter((c) => c.state === 'done').length
  const running    = chapters.filter((c) => c.state === 'running').length
  const err        = chapters.filter((c) => c.state === 'error').length
  const allChecked = sel.size === filtered.length && filtered.length > 0

  useEffect(() => {
    if (project) setHeader(project.title, [{ label: 'Dự án', to: '/projects' }])
    return () => clearHeader()
  }, [project, setHeader, clearHeader])

  if (pLoad) return (
    <div className="p-6 animate-pulse space-y-4">
      <div className="flex gap-5">
        <div className="w-32 h-44 rounded-xl bg-zinc-100 shrink-0" />
        <div className="flex-1 space-y-3 pt-2">
          <div className="h-7 w-48 rounded-lg bg-zinc-100" />
          <div className="h-4 w-64 rounded bg-zinc-100" />
          <div className="h-4 w-40 rounded bg-zinc-100" />
        </div>
      </div>
    </div>
  )

  if (pErr || !project) return (
    <div className="p-6">
      <p className="text-sm text-red-500 font-medium">Không tìm thấy dự án.</p>
      <Link to="/projects" className="text-sm text-zinc-500 underline mt-1 inline-block">← Quay lại</Link>
    </div>
  )

  return (
    <div>

      {/* hero */}
      <div className="px-6 pt-6 pb-5 flex items-start gap-5">
        <Cover
          src={project.cover_url}
          title={project.title}
          fontSize="text-2xl"
          className="w-32 h-44 rounded-xl border border-zinc-200 shrink-0"
        />

        <div className="flex-1 min-w-0 pt-1">
          <h1 className="text-2xl font-bold tracking-tight text-zinc-900 mb-2">
            {project.title}
          </h1>

          <div className="flex items-center gap-2 mb-3">
            <span className="inline-flex items-center h-7 px-3 rounded-full bg-zinc-100 text-sm text-zinc-600">
              {project.source_lang.toUpperCase() === 'JA' ? 'JP (日本語)' : project.source_lang.toUpperCase()}
            </span>
            <span className="text-xs text-zinc-400">→</span>
            <span className="inline-flex items-center h-7 px-3 rounded-full bg-zinc-100 text-sm text-zinc-600">
              {project.target_lang.toUpperCase() === 'VI' ? 'VI (Tiếng Việt)' : project.target_lang.toUpperCase()}
            </span>
          </div>

          <div className="flex items-center gap-5 text-sm text-zinc-400 flex-wrap">
            <span className="inline-flex items-center gap-1.5">
              <BookOpen size={13} className="shrink-0" />{total} chương
            </span>
            {project.updated_at && (
              <span className="inline-flex items-center gap-1.5">
                <Clock size={13} className="shrink-0" />Cập nhật: {timeAgo(project.updated_at)}
              </span>
            )}
          </div>

          {project.description && (
            <p className="mt-3 text-sm text-zinc-500 leading-relaxed line-clamp-3 max-w-2xl">
              {project.description}
            </p>
          )}

          {total > 0 && (
            <div className="flex items-center gap-4 mt-4 text-xs text-zinc-400">
              {done    > 0 && <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-green-500" />{done} hoàn thành</span>}
              {running > 0 && <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-blue-500" />{running} đang xử lý</span>}
              {err     > 0 && <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-red-500" />{err} lỗi</span>}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0 pt-1">
          <button className={btn.secondary}><Share2 size={14} />Chia sẻ</button>
          <button className={btn.secondary}><Download size={14} />Xuất</button>
          <button className={btn.icon}><MoreHorizontal size={15} /></button>
          <button className={btn.primary}><Plus size={14} />Thêm chương</button>
        </div>
      </div>

      {/* tabs */}
      <div className="flex items-center px-6">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={cn(
              'h-11 px-4 text-sm cursor-pointer transition-colors',
              tab === key
                ? 'text-zinc-900 font-semibold border-b-2 border-zinc-900 -mb-px'
                : 'text-zinc-400 hover:text-zinc-600 border-b border-zinc-200',
            )}
          >
            {label}
          </button>
        ))}
        <div className="flex-1 self-stretch border-b border-zinc-200" />
      </div>

      <div className="px-6 py-4">

        {tab === 'chapters' && (
          <>
            <div className="flex items-center justify-between mb-4">
              {/* iOS segmented control */}
              <div className="flex items-center bg-zinc-100 rounded-xl p-0.5">
                {FILTERS.map(({ key, label }) => (
                  <button
                    key={key}
                    onClick={() => setFilter(key)}
                    className={cn(
                      'h-7 px-3 rounded-xl text-sm transition-all cursor-pointer',
                      filter === key
                        ? 'bg-zinc-900 text-white font-medium shadow-[0_1px_3px_rgb(0,0,0,0.2)]'
                        : 'text-zinc-400 hover:text-zinc-600',
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2">
                <label className="flex items-center gap-2 h-8 px-3 w-44 rounded-lg border border-zinc-200 hover:border-zinc-300 transition-colors cursor-text outline-none">
                  <Search size={13} className="text-zinc-400 shrink-0" />
                  <input
                    type="text"
                    placeholder="Tìm chương..."
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-300"
                  />
                </label>
                <button className={cn(btn.icon, 'size-8')}>
                  <SlidersHorizontal size={14} />
                </button>
              </div>
            </div>

            <div className="rounded-xl border border-zinc-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-200">
                    <th className="w-10 px-4 py-3">
                      <button
                        onClick={toggleAll}
                        className={cn(
                          'size-4 rounded border flex items-center justify-center cursor-pointer transition-colors',
                          allChecked ? 'bg-blue-500 border-blue-500' : 'border-zinc-300 hover:border-zinc-400',
                        )}
                      >
                        {allChecked && <Check size={9} className="text-white" />}
                      </button>
                    </th>
                    {COLS.map(({ label, className }) => (
                      <th
                        key={label}
                        className={cn('px-3 py-3 text-left text-xs font-medium text-zinc-400', className)}
                      >
                        {label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cLoad && Array.from({ length: 5 }).map((_, i) => (
                    <tr key={i} className="border-b border-zinc-200 last:border-0">
                      <td colSpan={7} className="px-4 py-3">
                        <div className="h-3 rounded bg-zinc-100 animate-pulse" />
                      </td>
                    </tr>
                  ))}

                  {!cLoad && filtered.length === 0 && (
                    <tr>
                      <td colSpan={7} className="py-12 text-center text-sm text-zinc-400">
                        Không có chương nào
                      </td>
                    </tr>
                  )}

                  {!cLoad && filtered.map((ch) => {
                    const st = statusOf(ch.state)
                    const checked = sel.has(ch.chapter_id)
                    return (
                      <tr
                        key={ch.chapter_id}
                        className="border-b border-zinc-200 last:border-0 group"
                      >
                        <td className="px-4 py-3">
                          <button
                            onClick={() => toggleOne(ch.chapter_id)}
                            className={cn(
                              'size-4 rounded border flex items-center justify-center cursor-pointer transition-colors',
                              checked ? 'bg-blue-500 border-blue-500' : 'border-zinc-300 hover:border-zinc-400',
                            )}
                          >
                            {checked && <Check size={9} className="text-white" />}
                          </button>
                        </td>
                        <td className="px-3 py-3">
                          <span className="font-medium text-zinc-900">Chương {String(ch.idx).replace(/\./g, '-')}</span>
                          {ch.title && (
                            <span className="ml-2 text-xs text-zinc-400 truncate inline-block max-w-md align-middle">
                              {ch.title}
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-3 text-xs text-zinc-400 tabular-nums">
                          {ch.state === 'done'
                            ? `${ch.page_count}/${ch.page_count}`
                            : ch.progress
                            ? `${ch.progress.page_index}/${ch.page_count}`
                            : `0/${ch.page_count}`}
                        </td>
                        <td className="px-3 py-3">
                          <span className="inline-flex items-center gap-1.5">
                            <span className={cn('size-2 rounded-full shrink-0', st.dot)} />
                            <span className={cn('text-sm', st.text)}>{st.label}</span>
                          </span>
                        </td>
                        <td className="px-3 py-3 text-sm text-zinc-400 whitespace-nowrap">{timeAgo(ch.updated_at)}</td>
                        <td className="px-3 py-3">
                          <div className="flex items-center gap-1 transition-opacity">
                            <button className={btn.ghost}><Eye size={14} /></button>
                            <button className={btn.ghost}><Pencil size={14} /></button>
                            <button className={btn.ghost}><MoreHorizontal size={14} /></button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between mt-3 text-xs text-zinc-400">
              <span>Hiển thị {filtered.length} trong tổng số {total} chương</span>
              <button className="inline-flex items-center gap-1 hover:text-zinc-900 transition-colors cursor-pointer">
                Cuộn để xem thêm <ChevronDown size={11} />
              </button>
            </div>
          </>
        )}

        {tab === 'context' && <PipelineTab />}

      </div>

      {sel.size > 0 && (
        <div className="fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-white border border-zinc-200 rounded-2xl px-5 py-3 shadow-[0_8px_32px_rgb(0,0,0,0.12)]">
          <span className="text-sm text-zinc-500 tabular-nums">{sel.size} chương đã chọn</span>
          <button
            onClick={() => { alert(`Dịch ${sel.size} chương`); setSel(new Set()) }}
            className="inline-flex items-center gap-2 h-9 px-5 rounded-xl bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 transition-colors cursor-pointer"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M2 7h10M7 2l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Dịch {sel.size} chương đã chọn
          </button>
          <button
            onClick={() => setSel(new Set())}
            className="size-8 rounded-lg flex items-center justify-center text-zinc-300 hover:text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer"
          >
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
              <path d="M1 1L10 10M10 1L1 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      )}

    </div>
  )
}

export const Route = createFileRoute('/projects/$projectId')({
  component: ProjectDetailPage,
})
