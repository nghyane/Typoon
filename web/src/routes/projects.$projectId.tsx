import { createFileRoute, Link } from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState, useMemo, useEffect } from 'react'
import {
  Plus, Search, Share2, MoreHorizontal, Download,
  Check, Eye, RefreshCw, AlertCircle, Trash2,
  BookOpen, Clock, SlidersHorizontal, Settings, Sparkles,
} from 'lucide-react'
import { useHeaderStore } from '../store/header'
import { api, type ApiChapter, type ApiProject } from '../lib/api'
import { cn } from '../lib/cn'
import { timeAgo } from '../lib/time'
import { Cover } from '../components/Cover'
import { STATE, stageLabel, chapterStats, chapterPct } from '../lib/chapter'
import { btn } from '../components/ui'
import { toast } from '../components/Toaster'
import { PullFromUrlDialog } from '../components/PullFromUrlDialog'
import { GlossaryPanel } from '../components/GlossaryPanel'
import { SettingsPanel } from '../components/SettingsPanel'

// ── tabs / filters ────────────────────────────────────────────────────────────

type Tab = 'chapters' | 'glossary' | 'settings'

const TABS: { key: Tab; label: string; icon: typeof BookOpen }[] = [
  { key: 'chapters', label: 'Chương',     icon: BookOpen },
  { key: 'glossary', label: 'Thuật ngữ',  icon: Sparkles },
  { key: 'settings', label: 'Cài đặt',    icon: Settings },
]

type Filter = 'all' | 'idle' | 'running' | 'done' | 'error'

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all',     label: 'Tất cả'    },
  { key: 'running', label: 'Đang chạy' },
  { key: 'idle',    label: 'Chờ xử lý' },
  { key: 'done',    label: 'Xong'      },
  { key: 'error',   label: 'Lỗi'       },
]

function matchFilter(ch: ApiChapter, f: Filter): boolean {
  if (f === 'all')  return true
  if (f === 'idle') return ch.state === 'idle' || ch.state === 'pending'
  return ch.state === f
}

// ── chapter row ───────────────────────────────────────────────────────────────

function ChapterRow({
  ch, checked, onToggle, projectId,
}: {
  ch:        ApiChapter
  checked:   boolean
  onToggle:  () => void
  projectId: number
}) {
  const qc = useQueryClient()
  const st = STATE[ch.state]

  const redo = useMutation({
    mutationFn: () => api.redoChapter(projectId, ch.chapter_id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects', projectId, 'chapters'] }),
    onError:   (e: Error) => toast.error(e.message),
  })

  const del = useMutation({
    mutationFn: () => api.deleteChapter(projectId, ch.chapter_id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['projects', projectId, 'chapters'] })
      toast.success(`Đã xoá chương ${ch.idx}`)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const pct        = chapterPct(ch)
  const showBar    = ch.state === 'running' || ch.state === 'done'
  const stageText  = ch.stage ? stageLabel(ch.stage) : ''
  const subLabel   = ch.state === 'running' && stageText ? stageText : st.label

  return (
    <tr className="border-b border-zinc-100 last:border-0 group hover:bg-zinc-50/60 transition-colors">
      <td className="px-4 py-3">
        <button
          onClick={onToggle}
          className={cn(
            'size-4 rounded border flex items-center justify-center cursor-pointer transition-colors',
            checked ? 'bg-zinc-900 border-zinc-900' : 'border-zinc-300 hover:border-zinc-500',
          )}
        >
          {checked && <Check size={9} className="text-white" />}
        </button>
      </td>

      <td className="px-3 py-3 min-w-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="font-medium text-zinc-900 tabular-nums shrink-0">
            Ch.{String(ch.idx).replace(/\.0$/, '')}
          </span>
          {ch.title && (
            <span className="text-sm text-zinc-500 truncate">{ch.title}</span>
          )}
        </div>
        {ch.state === 'error' && ch.error && (
          <div className="mt-1 inline-flex items-center gap-1 text-xs text-red-500 truncate max-w-md">
            <AlertCircle size={11} className="shrink-0" />
            <span className="truncate" title={ch.error}>{ch.error}</span>
          </div>
        )}
      </td>

      <td className="px-3 py-3 w-56">
        <div className="flex items-center gap-2">
          <span className={cn('size-2 rounded-full shrink-0', st.dot)} />
          <span className={cn('text-xs', st.text)}>{subLabel}</span>
          {showBar && (
            <span className="text-xs text-zinc-400 tabular-nums ml-auto">
              {ch.state === 'done'
                ? `${ch.page_count}/${ch.page_count}`
                : ch.progress
                ? `${ch.progress.page_index}/${ch.progress.page_total}`
                : `0/${ch.page_count}`}
            </span>
          )}
        </div>
        {showBar && (
          <div className="h-1 mt-1.5 rounded-full bg-zinc-100 overflow-hidden">
            <div
              className={cn('h-full rounded-full transition-[width] duration-300', st.bar)}
              style={{ width: `${pct}%` }}
            />
          </div>
        )}
      </td>

      <td className="px-3 py-3 text-sm text-zinc-400 whitespace-nowrap w-32" title={ch.updated_at ?? ''}>
        {timeAgo(ch.updated_at)}
      </td>

      <td className="px-3 py-3 w-32">
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button className={btn.ghost} title="Xem render">
            <Eye size={14} />
          </button>
          <button
            className={btn.ghost}
            title="Chạy lại"
            disabled={redo.isPending || ch.state === 'running'}
            onClick={() => redo.mutate()}
          >
            <RefreshCw size={14} className={redo.isPending ? 'animate-spin' : ''} />
          </button>
          <button
            className={cn(btn.ghost, 'hover:text-red-600')}
            title="Xoá"
            disabled={del.isPending || ch.state === 'running'}
            onClick={() => {
              if (confirm(`Xoá chương ${ch.idx}?`)) del.mutate()
            }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </td>
    </tr>
  )
}

// ── page ──────────────────────────────────────────────────────────────────────

function ProjectDetailPage() {
  const { projectId } = Route.useParams()
  const id = Number(projectId)

  const [tab,    setTab]    = useState<Tab>('chapters')
  const [filter, setFilter] = useState<Filter>('all')
  const [q,      setQ]      = useState('')
  const [sel,    setSel]    = useState<Set<number>>(new Set())
  const [pullOpen, setPullOpen] = useState(false)
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)

  const { data: project, isLoading: pLoad, isError: pErr } = useQuery({
    queryKey: ['projects', id],
    queryFn:  () => api.getProject(id),
    enabled:  !isNaN(id),
  })

  const { data: chapters = [], isLoading: cLoad } = useQuery({
    queryKey: ['projects', id, 'chapters'],
    queryFn:  () => api.listChapters(id),
    enabled:  !isNaN(id) && tab === 'chapters',
  })

  const filtered = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return chapters.filter((c) => {
      if (!matchFilter(c, filter)) return false
      if (!needle) return true
      return (
        String(c.idx).includes(needle) ||
        (c.title?.toLowerCase().includes(needle) ?? false)
      )
    })
  }, [chapters, filter, q])

  const stats        = chapterStats(chapters)
  const allChecked   = sel.size === filtered.length && filtered.length > 0
  const existingNums = useMemo(() => new Set(chapters.map((c) => c.idx)), [chapters])

  useEffect(() => {
    if (project) setHeader(project.title, [{ label: 'Dự án', to: '/projects' }])
    return () => clearHeader()
  }, [project, setHeader, clearHeader])

  const toggleOne = (cid: number) =>
    setSel((prev) => {
      const next = new Set(prev)
      if (next.has(cid)) next.delete(cid)
      else next.add(cid)
      return next
    })

  const toggleAll = () =>
    sel.size === filtered.length && filtered.length > 0
      ? setSel(new Set())
      : setSel(new Set(filtered.map((c) => c.chapter_id)))

  if (pLoad) return (
    <div className="p-6 animate-pulse">
      <div className="flex gap-5">
        <div className="w-32 h-44 rounded-xl bg-zinc-100 shrink-0" />
        <div className="flex-1 space-y-3 pt-2">
          <div className="h-7 w-72 rounded-lg bg-zinc-100" />
          <div className="h-4 w-96 rounded bg-zinc-100" />
          <div className="h-4 w-80 rounded bg-zinc-100" />
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
      <Hero
        project={project}
        stats={stats}
        onAddChapters={() => setPullOpen(true)}
      />

      {/* tabs */}
      <div className="flex items-center px-6 border-b border-zinc-200">
        {TABS.map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={cn(
              'inline-flex items-center gap-2 h-11 px-4 text-sm cursor-pointer transition-colors',
              tab === key
                ? 'text-zinc-900 font-semibold border-b-2 border-zinc-900 -mb-px'
                : 'text-zinc-400 hover:text-zinc-700',
            )}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <div className="px-6 py-4">
        {tab === 'chapters' && (
          <ChaptersTab
            cLoad={cLoad}
            filtered={filtered}
            stats={stats}
            filter={filter} setFilter={setFilter}
            q={q}           setQ={setQ}
            sel={sel}       toggleOne={toggleOne} toggleAll={toggleAll}
            allChecked={allChecked}
            projectId={id}
            onPull={() => setPullOpen(true)}
          />
        )}
        {tab === 'glossary' && <GlossaryPanel projectId={id} />}
        {tab === 'settings' && <SettingsPanel project={project} />}
      </div>

      {sel.size > 0 && (
        <SelectionBar count={sel.size} onClear={() => setSel(new Set())} />
      )}

      <PullFromUrlDialog
        open={pullOpen}
        onClose={() => setPullOpen(false)}
        project={project}
        existing={existingNums}
      />
    </div>
  )
}

// ── hero ─────────────────────────────────────────────────────────────────────

function Hero({
  project, stats, onAddChapters,
}: {
  project: ApiProject
  stats:   ReturnType<typeof chapterStats>
  onAddChapters: () => void
}) {
  return (
    <div className="px-6 pt-6 pb-5 flex items-start gap-5">
      <Cover
        src={project.cover_url}
        title={project.title}
        fontSize="text-2xl"
        version={project.updated_at}
        className="w-32 h-44 rounded-xl border border-zinc-200 shrink-0 shadow-sm"
      />

      <div className="flex-1 min-w-0 pt-1">
        <h1 className="text-2xl font-bold tracking-tight text-zinc-900 line-clamp-2">
          {project.title}
        </h1>

        <div className="flex items-center gap-2 mt-2">
          <span className="inline-flex items-center h-6 px-2.5 rounded-full bg-zinc-100 text-xs font-medium text-zinc-600 uppercase tracking-wide">
            {project.source_lang}
          </span>
          <span className="text-xs text-zinc-300">→</span>
          <span className="inline-flex items-center h-6 px-2.5 rounded-full bg-zinc-900 text-xs font-medium text-white uppercase tracking-wide">
            {project.target_lang}
          </span>
        </div>

        <div className="flex items-center gap-5 text-xs text-zinc-400 mt-3 flex-wrap">
          <span className="inline-flex items-center gap-1.5">
            <BookOpen size={12} />{stats.total} chương
          </span>
          {project.updated_at && (
            <span className="inline-flex items-center gap-1.5">
              <Clock size={12} />Cập nhật {timeAgo(project.updated_at)}
            </span>
          )}
        </div>

        {project.description && (
          <p className="mt-3 text-sm text-zinc-500 leading-relaxed line-clamp-3 max-w-2xl">
            {project.description}
          </p>
        )}

        {stats.total > 0 && (
          <div className="flex items-center gap-4 mt-4 text-xs">
            {stats.done    > 0 && <Pill dot="bg-emerald-500" label={`${stats.done} hoàn thành`} />}
            {stats.running > 0 && <Pill dot="bg-blue-500"    label={`${stats.running} đang xử lý`} />}
            {stats.error   > 0 && <Pill dot="bg-red-500"     label={`${stats.error} lỗi`} />}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 shrink-0 pt-1">
        <button className={btn.secondary}><Share2 size={14} />Chia sẻ</button>
        <button className={btn.secondary}><Download size={14} />Xuất</button>
        <button className={btn.iconBox}><MoreHorizontal size={15} /></button>
        <button className={btn.primary} onClick={onAddChapters}>
          <Plus size={14} />Pull từ URL
        </button>
      </div>
    </div>
  )
}

// ── chapters tab ─────────────────────────────────────────────────────────────

interface ChaptersTabProps {
  cLoad:      boolean
  filtered:   ApiChapter[]
  stats:      ReturnType<typeof chapterStats>
  filter:     Filter; setFilter: (f: Filter) => void
  q:          string; setQ: (s: string) => void
  sel:        Set<number>
  toggleOne:  (id: number) => void
  toggleAll:  () => void
  allChecked: boolean
  projectId:  number
  onPull:     () => void
}

function ChaptersTab({
  cLoad, filtered, stats, filter, setFilter, q, setQ,
  sel, toggleOne, toggleAll, allChecked, projectId, onPull,
}: ChaptersTabProps) {
  return (
    <>
      <div className="flex items-center justify-between gap-3 mb-4">
        <Segmented value={filter} onChange={setFilter} stats={stats} />
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 h-8 px-3 w-52 rounded-lg border border-zinc-200 hover:border-zinc-300 focus-within:border-zinc-400 transition-colors cursor-text">
            <Search size={13} className="text-zinc-400 shrink-0" />
            <input
              type="text"
              placeholder="Tìm chương..."
              value={q}
              onChange={(e) => setQ(e.target.value)}
              className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-300"
            />
          </label>
          <button className={cn(btn.iconBox, 'size-8')} title="Bộ lọc">
            <SlidersHorizontal size={14} />
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-zinc-200 overflow-hidden bg-white">
        <table className="w-full text-sm table-fixed">
          <thead>
            <tr className="border-b border-zinc-100 bg-zinc-50/50">
              <th className="w-10 px-4 py-2.5">
                <button
                  onClick={toggleAll}
                  disabled={filtered.length === 0}
                  className={cn(
                    'size-4 rounded border flex items-center justify-center cursor-pointer transition-colors disabled:opacity-30 disabled:cursor-not-allowed',
                    allChecked ? 'bg-zinc-900 border-zinc-900' : 'border-zinc-300 hover:border-zinc-500',
                  )}
                >
                  {allChecked && <Check size={9} className="text-white" />}
                </button>
              </th>
              <Th>Chương</Th>
              <Th className="w-56">Tiến độ</Th>
              <Th className="w-32">Cập nhật</Th>
              <Th className="w-32">Thao tác</Th>
            </tr>
          </thead>
          <tbody>
            {cLoad && Array.from({ length: 6 }).map((_, i) => (
              <tr key={i} className="border-b border-zinc-100 last:border-0">
                <td colSpan={5} className="px-4 py-3.5">
                  <div className="h-3 rounded bg-zinc-100 animate-pulse" />
                </td>
              </tr>
            ))}

            {!cLoad && filtered.length === 0 && (
              <tr>
                <td colSpan={5} className="py-16 text-center">
                  <p className="text-sm text-zinc-500 font-medium">
                    {stats.total === 0 ? 'Chưa có chương nào' : 'Không có chương nào'}
                  </p>
                  <p className="text-xs text-zinc-400 mt-1">
                    {stats.total === 0
                      ? 'Pull từ URL để bắt đầu dịch'
                      : (q || filter !== 'all' ? 'Thử bỏ bộ lọc' : 'Thêm chương để bắt đầu')}
                  </p>
                  {stats.total === 0 && (
                    <button
                      onClick={onPull}
                      className="mt-4 inline-flex items-center gap-1.5 h-8 px-4 rounded-lg bg-zinc-900 text-white text-xs font-medium hover:bg-zinc-700 cursor-pointer"
                    >
                      <Plus size={12} />
                      Pull từ URL
                    </button>
                  )}
                </td>
              </tr>
            )}

            {!cLoad && filtered.map((ch) => (
              <ChapterRow
                key={ch.chapter_id}
                ch={ch}
                projectId={projectId}
                checked={sel.has(ch.chapter_id)}
                onToggle={() => toggleOne(ch.chapter_id)}
              />
            ))}
          </tbody>
        </table>
      </div>

      {!cLoad && filtered.length > 0 && (
        <p className="text-xs text-zinc-400 mt-3">
          Hiển thị {filtered.length} trong {stats.total} chương
        </p>
      )}
    </>
  )
}

// ── small components ──────────────────────────────────────────────────────────

function Pill({ dot, label }: { dot: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-zinc-500">
      <span className={cn('size-2 rounded-full', dot)} />
      {label}
    </span>
  )
}

function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <th className={cn('px-3 py-2.5 text-left text-xs font-medium text-zinc-400', className)}>
      {children}
    </th>
  )
}

function Segmented({
  value, onChange, stats,
}: {
  value:    Filter
  onChange: (f: Filter) => void
  stats:    ReturnType<typeof chapterStats>
}) {
  const counts: Record<Filter, number> = {
    all:     stats.total,
    running: stats.running,
    idle:    stats.total - stats.done - stats.running - stats.error,
    done:    stats.done,
    error:   stats.error,
  }
  return (
    <div className="flex items-center bg-zinc-100 rounded-lg p-0.5">
      {FILTERS.map(({ key, label }) => {
        const n = counts[key]
        return (
          <button
            key={key}
            onClick={() => onChange(key)}
            className={cn(
              'h-7 px-3 rounded-md text-xs cursor-pointer transition-all',
              value === key
                ? 'bg-white text-zinc-900 font-medium shadow-sm'
                : 'text-zinc-500 hover:text-zinc-900',
            )}
          >
            {label}
            {n > 0 && <span className="ml-1.5 tabular-nums text-zinc-400">{n}</span>}
          </button>
        )
      })}
    </div>
  )
}

function SelectionBar({ count, onClear }: { count: number; onClear: () => void }) {
  return (
    <div className="fixed bottom-5 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 bg-white border border-zinc-200 rounded-2xl pl-5 pr-3 py-2.5 shadow-[0_8px_32px_rgb(0,0,0,0.12)]">
      <span className="text-sm text-zinc-500 tabular-nums">
        {count} chương đã chọn
      </span>
      <button
        onClick={() => { alert(`Dịch ${count} chương`); onClear() }}
        className="inline-flex items-center gap-2 h-9 px-4 rounded-xl bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 transition-colors cursor-pointer"
      >
        Dịch lại
      </button>
      <button
        onClick={onClear}
        className="size-8 rounded-lg flex items-center justify-center text-zinc-300 hover:text-zinc-600 hover:bg-zinc-100 transition-colors cursor-pointer"
      >
        ✕
      </button>
    </div>
  )
}

export const Route = createFileRoute('/projects/$projectId')({
  component: ProjectDetailPage,
})
