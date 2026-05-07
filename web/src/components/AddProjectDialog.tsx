import { useState, useMemo, useEffect } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Search, Globe, Check } from 'lucide-react'
import { api, type ApiSourceInfo } from '../lib/api'
import { cn } from '../lib/cn'
import { Modal } from './Modal'
import { Cover } from './Cover'
import { btn, input, label, Spinner } from './ui'
import { toast } from './Toaster'

interface Props { open: boolean; onClose: () => void }

const TARGET_LANGS = [
  { code: 'vi', label: 'Tiếng Việt' },
  { code: 'en', label: 'English'    },
  { code: 'zh', label: '中文'        },
  { code: 'ja', label: '日本語'      },
]

export function AddProjectDialog({ open, onClose }: Props) {
  const qc = useQueryClient()
  const nav = useNavigate()

  const [url,      setUrl]      = useState('')
  const [info,     setInfo]     = useState<ApiSourceInfo | null>(null)
  const [target,   setTarget]   = useState('vi')
  const [selected, setSelected] = useState<Set<number>>(new Set())

  // Reset on close
  useEffect(() => {
    if (!open) {
      setUrl('')
      setInfo(null)
      setSelected(new Set())
    }
  }, [open])

  const discover = useMutation({
    mutationFn: () => api.discover(url.trim()),
    onSuccess: (data) => {
      setInfo(data)
      // Default: select all chapters
      setSelected(new Set(data.chapters.map((c) => c.number)))
    },
    onError: (e: Error) => toast.error(`Không quét được: ${e.message}`),
  })

  const pull = useMutation({
    mutationFn: () => api.pullNew({
      url:         url.trim(),
      target_lang: target,
      chapters:    [...selected].sort((a, b) => a - b),
    }),
    onSuccess: (proj) => {
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success(`Đã tạo dự án: ${proj.title}`)
      onClose()
      nav({ to: '/projects/$projectId', params: { projectId: String(proj.project_id) } })
    },
    onError: (e: Error) => toast.error(`Tạo thất bại: ${e.message}`),
  })

  const stats = useMemo(() => {
    if (!info) return null
    return {
      total: info.chapters.length,
      min:   info.chapters[0]?.number,
      max:   info.chapters.at(-1)?.number,
    }
  }, [info])

  const allSelected = info && selected.size === info.chapters.length && info.chapters.length > 0

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Thêm dự án từ URL"
      size="lg"
      footer={
        <>
          <button onClick={onClose} className={btn.secondary}>Huỷ</button>
          {info ? (
            <button
              onClick={() => pull.mutate()}
              disabled={pull.isPending || selected.size === 0}
              className={btn.primary}
            >
              {pull.isPending && <Spinner />}
              Tạo dự án ({selected.size} chương)
            </button>
          ) : (
            <button
              onClick={() => discover.mutate()}
              disabled={discover.isPending || !url.trim()}
              className={btn.primary}
            >
              {discover.isPending && <Spinner />}
              <Search size={14} />
              Quét
            </button>
          )}
        </>
      }
    >
      <div className="px-5 py-4 space-y-4">
        {/* URL field */}
        <div>
          <label className={label}>Nguồn</label>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Globe size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
              <input
                type="url"
                placeholder="https://comix.to/manga/..."
                value={url}
                onChange={(e) => { setUrl(e.target.value); setInfo(null) }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !info && url.trim() && !discover.isPending) {
                    discover.mutate()
                  }
                }}
                disabled={discover.isPending}
                className={cn(input, 'pl-9')}
                autoFocus
              />
            </div>
          </div>
        </div>

        {/* Loading skeleton */}
        {discover.isPending && (
          <div className="flex items-start gap-4 p-4 rounded-xl border border-zinc-200 bg-zinc-50/40">
            <div className="w-20 h-28 rounded-lg bg-zinc-100 animate-pulse" />
            <div className="flex-1 space-y-2 pt-1">
              <div className="h-5 w-3/4 rounded bg-zinc-100 animate-pulse" />
              <div className="h-3 w-1/2 rounded bg-zinc-100 animate-pulse" />
              <div className="h-3 w-2/3 rounded bg-zinc-100 animate-pulse" />
            </div>
          </div>
        )}

        {/* Result preview */}
        {info && stats && (
          <>
            <div className="flex items-start gap-4 p-4 rounded-xl border border-zinc-200 bg-zinc-50/40">
              <Cover
                src={info.cover_url}
                title={info.suggested_title}
                className="w-20 h-28 rounded-lg shrink-0"
                fontSize="text-lg"
              />
              <div className="flex-1 min-w-0">
                <h3 className="text-base font-semibold text-zinc-900 leading-snug line-clamp-2">
                  {info.suggested_title}
                </h3>
                <p className="text-xs text-zinc-500 mt-1">
                  {stats.total} chương ({stats.min}–{stats.max}) · {info.source_lang.toUpperCase()} → {target.toUpperCase()}
                </p>
                {info.description && (
                  <p className="text-xs text-zinc-500 mt-2 line-clamp-3">{info.description}</p>
                )}
              </div>
            </div>

            {/* Target lang */}
            <div>
              <label className={label}>Dịch sang</label>
              <div className="flex gap-1.5">
                {TARGET_LANGS.map((l) => (
                  <button
                    key={l.code}
                    onClick={() => setTarget(l.code)}
                    className={cn(
                      'h-8 px-3 rounded-lg text-xs cursor-pointer transition-colors border',
                      target === l.code
                        ? 'bg-zinc-900 text-white border-zinc-900 font-medium'
                        : 'bg-white text-zinc-600 border-zinc-200 hover:border-zinc-300',
                    )}
                  >
                    {l.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Chapter selector */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className={label} style={{ marginBottom: 0 }}>
                  Chọn chương ({selected.size}/{stats.total})
                </label>
                <div className="flex gap-3 text-xs">
                  <button
                    onClick={() => setSelected(new Set(info.chapters.map((c) => c.number)))}
                    className="text-zinc-500 hover:text-zinc-900 cursor-pointer"
                  >
                    Tất cả
                  </button>
                  <button
                    onClick={() => setSelected(new Set())}
                    className="text-zinc-500 hover:text-zinc-900 cursor-pointer"
                  >
                    Bỏ chọn
                  </button>
                </div>
              </div>
              <ChapterGrid
                chapters={info.chapters}
                selected={selected}
                onToggle={(n) => setSelected((p) => {
                  const next = new Set(p)
                  if (next.has(n)) next.delete(n)
                  else next.add(n)
                  return next
                })}
                allSelected={!!allSelected}
              />
            </div>
          </>
        )}
      </div>
    </Modal>
  )
}

function ChapterGrid({
  chapters, selected, onToggle, allSelected,
}: {
  chapters:    ApiSourceInfo['chapters']
  selected:    Set<number>
  onToggle:    (n: number) => void
  allSelected: boolean
}) {
  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(72px,1fr))] gap-1.5 max-h-72 overflow-auto p-2 rounded-lg border border-zinc-200 bg-zinc-50/30">
      {chapters.map((c) => {
        const on = selected.has(c.number)
        return (
          <button
            key={c.number}
            onClick={() => onToggle(c.number)}
            className={cn(
              'relative h-10 rounded-md text-xs font-medium tabular-nums cursor-pointer transition-colors border',
              on
                ? 'bg-zinc-900 text-white border-zinc-900'
                : 'bg-white text-zinc-600 border-zinc-200 hover:border-zinc-400',
            )}
            title={c.title ?? `Chương ${c.number}`}
          >
            {on && <Check size={10} className="absolute top-1 right-1 text-white/80" />}
            {String(c.number).replace(/\.0$/, '')}
          </button>
        )
      })}
      {allSelected && chapters.length === 0 && (
        <p className="col-span-full text-xs text-zinc-400 text-center py-4">Không có chương</p>
      )}
    </div>
  )
}
