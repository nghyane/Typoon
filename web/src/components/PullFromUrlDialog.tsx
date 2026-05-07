import { useState, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Search, Globe, Check, ChevronRight } from 'lucide-react'
import { api, type ApiProject, type ApiSourceConnector, type ApiSourceInfo } from '../lib/api'
import { cn } from '../lib/cn'
import { Modal } from './Modal'
import { Cover } from './Cover'
import { btn, input, label, Spinner } from './ui'
import { toast } from './Toaster'

interface Props {
  open:    boolean
  onClose: () => void
  project: ApiProject
  /** Chapter numbers already in the project — pre-disabled. */
  existing: Set<number>
}

export function PullFromUrlDialog({ open, onClose, project, existing }: Props) {
  const qc = useQueryClient()

  const { data: connectors = [] } = useQuery({
    queryKey: ['sources'],
    queryFn:  api.listSources,
    staleTime: 60_000,
  })

  const [url,      setUrl]      = useState(project.source_url ?? '')
  const [info,     setInfo]     = useState<ApiSourceInfo | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())

  useEffect(() => {
    if (open) setUrl(project.source_url ?? '')
    if (!open) {
      setInfo(null)
      setSelected(new Set())
    }
  }, [open, project.source_url])

  const matchedConnector = useMemo<ApiSourceConnector | null>(() => {
    if (!url.trim()) return null
    return connectors.find((c) => url.includes(c.id)) ?? null
  }, [url, connectors])

  const discover = useMutation({
    mutationFn: () => api.discover(url.trim()),
    onSuccess: (data) => {
      setInfo(data)
      // Default: only NEW chapters (not already in the project).
      setSelected(new Set(
        data.chapters
          .filter((c) => !existing.has(c.number))
          .map((c) => c.number),
      ))
    },
    onError: (e: Error) => toast.error(`Không quét được: ${e.message}`),
  })

  const pull = useMutation({
    mutationFn: () => api.pull(project.project_id, {
      url:      url.trim(),
      chapters: [...selected].sort((a, b) => a - b),
    }),
    onSuccess: ({ queued }) => {
      qc.invalidateQueries({ queryKey: ['projects', project.project_id] })
      qc.invalidateQueries({ queryKey: ['projects', project.project_id, 'chapters'] })
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success(`Đã thêm ${queued} chương vào hàng đợi`)
      onClose()
    },
    onError: (e: Error) => toast.error(`Thêm thất bại: ${e.message}`),
  })

  const newCount = info
    ? info.chapters.filter((c) => !existing.has(c.number)).length
    : 0

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Pull từ URL"
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
              Tải {selected.size} chương
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
        {/* Site catalog */}
        {!info && (
          <div>
            <label className={label}>Site hỗ trợ</label>
            <div className="space-y-1.5">
              {connectors.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setUrl(c.example_url)}
                  className={cn(
                    'w-full flex items-center gap-3 p-3 rounded-lg border text-left cursor-pointer transition-colors',
                    matchedConnector?.id === c.id
                      ? 'border-zinc-900 bg-zinc-50'
                      : 'border-zinc-200 bg-white hover:border-zinc-300',
                  )}
                >
                  <div className="size-8 rounded-md bg-zinc-100 flex items-center justify-center shrink-0">
                    <Globe size={14} className="text-zinc-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-zinc-900">{c.name}</span>
                      <span className="text-[11px] uppercase tracking-wide text-zinc-400">{c.source_lang}</span>
                    </div>
                    <p className="text-xs text-zinc-500 truncate">{c.description}</p>
                  </div>
                  <ChevronRight size={14} className="text-zinc-300" />
                </button>
              ))}
              {connectors.length === 0 && (
                <p className="text-xs text-zinc-400">Không có connector nào.</p>
              )}
            </div>
          </div>
        )}

        {/* URL */}
        <div>
          <label className={label}>URL</label>
          <div className="relative">
            <Globe size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
            <input
              type="url"
              placeholder={matchedConnector?.example_url ?? 'https://...'}
              value={url}
              onChange={(e) => { setUrl(e.target.value); setInfo(null) }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !info && url.trim() && !discover.isPending) {
                  discover.mutate()
                }
              }}
              disabled={discover.isPending}
              className={cn(input, 'pl-9')}
            />
          </div>
          {matchedConnector && !info && (
            <p className="text-xs text-zinc-400 mt-1">
              Khớp connector: <span className="text-zinc-600 font-medium">{matchedConnector.name}</span>
            </p>
          )}
        </div>

        {/* Result preview + chapter picker */}
        {info && (
          <>
            <div className="flex items-start gap-4 p-4 rounded-xl border border-zinc-200 bg-zinc-50/40">
              <Cover
                src={info.cover_url}
                title={info.suggested_title}
                className="w-16 h-22 rounded-lg shrink-0"
                fontSize="text-base"
              />
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-semibold text-zinc-900 leading-snug line-clamp-2">
                  {info.suggested_title}
                </h3>
                <p className="text-xs text-zinc-500 mt-1">
                  {info.chapters.length} chương · {info.source_lang.toUpperCase()}
                </p>
                {info.description && (
                  <p className="text-xs text-zinc-500 mt-1.5 line-clamp-2">{info.description}</p>
                )}
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-2 text-xs">
                <span className="text-zinc-500">
                  <span className="text-zinc-900 font-medium tabular-nums">{newCount}</span> chương mới ·
                  {' '}
                  <span className="text-zinc-400 tabular-nums">{existing.size} đã có</span>
                </span>
                <div className="flex gap-3">
                  <button
                    onClick={() =>
                      setSelected(new Set(
                        info.chapters
                          .filter((c) => !existing.has(c.number))
                          .map((c) => c.number),
                      ))
                    }
                    className="text-zinc-500 hover:text-zinc-900 cursor-pointer"
                  >
                    Chỉ chương mới
                  </button>
                  <button
                    onClick={() => setSelected(new Set(info.chapters.map((c) => c.number).filter((n) => !existing.has(n))))}
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

              <div className="grid grid-cols-[repeat(auto-fill,minmax(72px,1fr))] gap-1.5 max-h-72 overflow-auto p-2 rounded-lg border border-zinc-200 bg-zinc-50/30">
                {info.chapters.map((c) => {
                  const own = existing.has(c.number)
                  const on  = selected.has(c.number)
                  return (
                    <button
                      key={c.number}
                      onClick={() => {
                        if (own) return
                        setSelected((p) => {
                          const next = new Set(p)
                          if (next.has(c.number)) next.delete(c.number)
                          else next.add(c.number)
                          return next
                        })
                      }}
                      disabled={own}
                      title={own ? 'Đã có trong dự án' : c.title ?? `Chương ${c.number}`}
                      className={cn(
                        'relative h-10 rounded-md text-xs font-medium tabular-nums cursor-pointer transition-colors border',
                        own
                          ? 'bg-zinc-100 text-zinc-300 border-zinc-100 cursor-not-allowed'
                          : on
                          ? 'bg-zinc-900 text-white border-zinc-900'
                          : 'bg-white text-zinc-600 border-zinc-200 hover:border-zinc-400',
                      )}
                    >
                      {on && !own && <Check size={10} className="absolute top-1 right-1 text-white/80" />}
                      {String(c.number).replace(/\.0$/, '')}
                    </button>
                  )
                })}
              </div>
            </div>
          </>
        )}
      </div>
    </Modal>
  )
}
