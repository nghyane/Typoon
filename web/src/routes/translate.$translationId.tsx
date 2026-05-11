import { createFileRoute, Link, useNavigate } from '@tanstack/react-router'
import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ArrowLeft, BookOpen, Sparkles, RefreshCw, Trash2, Share2, Lock,
  AlertCircle, CheckCircle2, Loader2, RotateCcw,
} from 'lucide-react'
import { Cover } from '@shared/ui/Cover'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { useHeaderStore } from '../store/header'
import {
  api,
  type ApiBubbleEdit, type ApiMyTranslation, type ApiTranslation,
} from '@shared/api/api'
import { useTranslateProgress } from '@features/translate/useTranslateProgress'

// =============================================================================
// /translate/$translationId — single-translation editor.
//
// Layout (pro-grade, mirrors the old ProjectDetail surface):
//
//   ┌─ Header ──────────────────────────────────────────────────────┐
//   │ ← Back     Naruto · Ch 1099 · 🇻🇳 Tiếng Việt                  │
//   │            ●done · last update 2h trước                        │
//   │ [Đọc] [Render lại] [Bật chia sẻ] [Xoá]                         │
//   ╰────────────────────────────────────────────────────────────────╯
//   ┌─ Page ────────────────┐  ┌─ Bubble panel ────────────────────┐
//   │ Bubble row list:      │  │ Source: ナルトだ!                  │
//   │  1 (selected)         │  │ Draft : Tao là Naruto!            │
//   │  2                    │  │ Edit  : [ ... ]                   │
//   │  3 (edited)           │  │ [Lưu sửa] [Khôi phục]              │
//   │  ...                  │  ╰────────────────────────────────────╯
//   ╰───────────────────────╯
//
// The chapter reader (with the visual page + bubble overlay) opens in
// /browse/.../chapter/$id and stays the dedicated reading surface;
// /translate/{id} is for text editing of bubbles + render management.
// =============================================================================

interface SearchParams {
  page?: number
}

function TranslationEditorPage() {
  const { translationId } = Route.useParams()
  const id = Number(translationId)
  const nav = useNavigate()

  const { data: t, isPending, isError } = useQuery({
    queryKey: ['translation', id],
    queryFn:  () => api.getTranslation(id),
    enabled:  Number.isInteger(id) && id > 0,
    refetchInterval: (q) => {
      // Poll while not done — SSE supplies more granular signal but
      // a low-frequency poll keeps the archive_url in sync after the
      // render task lands.
      const data = q.state.data
      return data && data.state === 'done' ? false : 5_000
    },
  })

  // Resolve the chapter + material context once we have a translation.
  // We refetch /api/translate/mine and pick our row — cheap because
  // the editor opens from /translate/ where the list is already in
  // cache.
  const { data: mineRows = [] } = useQuery({
    queryKey: ['translations', 'mine'],
    queryFn:  api.listMyTranslations,
    staleTime: 30_000,
  })
  const meta = useMemo(
    () => mineRows.find((r) => r.translation_id === id) ?? null,
    [mineRows, id],
  )

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader(
      meta ? `${meta.material_title} · Ch ${meta.chapter_number}` : 'Bản dịch',
      [{ label: 'Bản dịch của tôi', to: '/translate' }],
    )
    return () => clearHeader()
  }, [meta, setHeader, clearHeader])

  // Live progress while pipeline runs.
  const progress = useTranslateProgress(id, t?.state !== 'done')

  if (isPending || !t) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (isError) {
    return (
      <EmptyState
        icon={AlertCircle}
        title="Không tải được bản dịch"
      />
    )
  }

  return (
    <div className="pb-16">
      <Header
        translation={t}
        meta={meta}
        progress={progress?.state ?? null}
        onDeleted={() => nav({ to: '/translate' })}
      />

      <main className="mt-4 px-4 sm:px-6">
        {t.state === 'done' ? (
          <BubbleEditor translationId={t.id} />
        ) : (
          <PipelineProgress
            translationId={t.id}
            currentStage={progress?.stage ?? ''}
            index={progress?.index ?? 0}
            total={progress?.total ?? 0}
            errorMessage={progress?.error}
            state={progress?.state ?? null}
          />
        )}
      </main>
    </div>
  )
}

// ── Header strip ─────────────────────────────────────────────────────

function Header({
  translation, meta, onDeleted,
}: {
  translation: ApiTranslation
  meta:        ApiMyTranslation | null
  progress:    string | null
  onDeleted:   () => void
}) {
  const qc = useQueryClient()
  const t  = translation

  const redo = useMutation({
    mutationFn: () => api.redoTranslation(t.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['translation', t.id] })
    },
  })

  const toggleShare = useMutation({
    mutationFn: () => api.patchTranslation(t.id, { in_feed: !t.in_feed }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['translation', t.id] })
    },
  })

  const del = useMutation({
    mutationFn: () => api.deleteTranslation(t.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['translations', 'mine'] })
      onDeleted()
    },
  })

  return (
    <header className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4 border-b border-border-soft">
      <Link
        to="/translate"
        className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text mb-3"
      >
        <ArrowLeft size={14} />
        Bản dịch của tôi
      </Link>

      <div className="flex items-start gap-4">
        {meta?.material_cover ? (
          <Cover
            src={meta.material_cover}
            title={meta.material_title}
            className="size-16 rounded-sm shrink-0"
          />
        ) : null}

        <div className="flex-1 min-w-0">
          <h1 className="text-base sm:text-lg font-semibold text-text leading-tight">
            {meta?.material_title ?? '—'}
            <span className="text-text-subtle font-normal ml-2">
              · Ch {meta?.chapter_number ?? '?'}
            </span>
          </h1>
          <p className="text-xs text-text-subtle mt-1 flex items-center gap-2">
            <span className="uppercase font-semibold text-text">{t.target_lang}</span>
            <StateBadge state={t.state} />
            {t.has_edits && (
              <span className="text-amber-400">Đã sửa</span>
            )}
          </p>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {t.archive_url && (
              <a
                href={t.archive_url}
                target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 h-8 px-3 rounded-sm bg-accent text-accent-fg text-[13px] font-medium hover:bg-accent-hover cursor-pointer"
              >
                <BookOpen size={13} />
                Đọc
              </a>
            )}
            <Button
              size="sm" variant="secondary"
              onClick={() => redo.mutate()}
              disabled={redo.isPending}
            >
              <RefreshCw size={12} />
              Render lại
            </Button>
            <Button
              size="sm" variant="secondary"
              onClick={() => toggleShare.mutate()}
              disabled={toggleShare.isPending}
            >
              {t.in_feed ? <Lock size={12} /> : <Share2 size={12} />}
              {t.in_feed ? 'Bỏ chia sẻ' : 'Chia sẻ'}
            </Button>
            <Button
              size="sm" variant="danger"
              onClick={() => {
                if (confirm('Xoá bản dịch này? Không thể khôi phục.')) {
                  del.mutate()
                }
              }}
              disabled={del.isPending}
            >
              <Trash2 size={12} />
              Xoá
            </Button>
          </div>
        </div>
      </div>
    </header>
  )
}

function StateBadge({ state }: { state: string }) {
  if (state === 'done')    return <span className="text-success-text">● Xong</span>
  if (state === 'running') return <span className="text-accent-text">● Đang chạy</span>
  if (state === 'error')   return <span className="text-error-text">● Lỗi</span>
  return <span className="text-text-muted">● Chờ</span>
}

// ── Pipeline progress (when state != done) ───────────────────────────

function PipelineProgress({
  currentStage, index, total, errorMessage, state,
}: {
  translationId: number
  currentStage:  string
  index:         number
  total:         number
  errorMessage?: string
  state:         string | null
}) {
  const stages: { id: string; label: string; cost: string }[] = [
    { id: 'prepare',   label: 'Chuẩn bị ảnh',      cost: '~5s' },
    { id: 'scan',      label: 'OCR + nhận diện',   cost: '1-3 phút' },
    { id: 'translate', label: 'Dịch bằng LLM',     cost: '~1 phút' },
    { id: 'render',    label: 'Render trang dịch', cost: '~30s' },
  ]
  const currentIdx = stages.findIndex((s) => s.id === currentStage)

  if (state === 'error') {
    return (
      <div className="rounded-md bg-error/10 ring-1 ring-inset ring-error/30 px-4 py-3 text-sm">
        <p className="font-medium text-error-text mb-1">Pipeline lỗi</p>
        <p className="text-text-muted text-[13px]">
          {errorMessage ?? 'Không có thông báo chi tiết.'}
        </p>
        <p className="mt-3 text-[12px] text-text-subtle">
          Bấm "Render lại" ở trên để chạy lại từ đầu.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-md bg-surface p-4">
      <p className="text-sm text-text font-medium mb-3 flex items-center gap-2">
        <Loader2 size={14} className="animate-spin text-accent-text" />
        Đang xử lý
      </p>
      <ol className="space-y-1.5">
        {stages.map((s, i) => {
          const done    = currentIdx > i
          const active  = currentIdx === i
          const pct     = active && total > 0
            ? Math.min(100, Math.round((index / total) * 100))
            : null
          return (
            <li key={s.id} className="flex items-center gap-3 text-[13px]">
              <span className={cn(
                'size-5 rounded-full flex items-center justify-center shrink-0',
                done   ? 'bg-success text-bg' :
                active ? 'bg-accent text-accent-fg' :
                         'bg-surface-2 text-text-subtle',
              )}>
                {done ? <CheckCircle2 size={11} /> :
                 active ? <Loader2 size={11} className="animate-spin" /> :
                          <span className="size-1 rounded-full bg-current" />}
              </span>
              <span className={cn(
                'flex-1',
                active ? 'text-text font-medium' :
                done   ? 'text-text-muted' :
                         'text-text-subtle',
              )}>
                {s.label}
              </span>
              <span className="text-[11px] text-text-subtle tabular shrink-0">
                {active && pct != null ? `${pct}%` : s.cost}
              </span>
            </li>
          )
        })}
      </ol>
    </div>
  )
}

// ── Bubble editor (when state == done) ───────────────────────────────

function BubbleEditor({ translationId }: { translationId: number }) {
  const { data: bubbles = [], isPending } = useQuery({
    queryKey: ['translation', translationId, 'bubbles'],
    queryFn:  () => api.listTranslationBubbles(translationId),
    staleTime: 60_000,
  })

  if (isPending) {
    return (
      <div className="flex items-center justify-center py-12">
        <Spinner size={16} />
      </div>
    )
  }

  if (bubbles.length === 0) {
    return (
      <EmptyState
        icon={Sparkles}
        title="Chương không có bubble"
        hint="OCR không tìm thấy text. Có thể đây là chapter trống hoặc full-page art."
      />
    )
  }

  // Group bubbles by page for visual breakdown.
  const pages: Map<number, ApiBubbleEdit[]> = new Map()
  for (const b of bubbles) {
    if (!pages.has(b.page_index)) pages.set(b.page_index, [])
    pages.get(b.page_index)!.push(b)
  }

  return (
    <div className="space-y-6">
      {[...pages.entries()].map(([pageIdx, pageBubbles]) => (
        <section key={pageIdx}>
          <h2 className="text-[11px] uppercase tracking-wider text-text-subtle mb-2">
            Trang {pageIdx + 1}
          </h2>
          <ul className="rounded-md bg-surface divide-y divide-border-soft overflow-hidden">
            {pageBubbles.map((b) => (
              <BubbleRow
                key={`${b.page_index}::${b.bubble_idx}`}
                translationId={translationId}
                bubble={b}
              />
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

function BubbleRow({
  translationId, bubble,
}: {
  translationId: number
  bubble:        ApiBubbleEdit
}) {
  const qc = useQueryClient()
  const initial = bubble.edited_text ?? bubble.draft_text
  const [draft, setDraft] = useState(initial)
  const [editing, setEditing] = useState(false)

  // Sync when server bubble changes underneath us (e.g. after redo).
  useEffect(() => { setDraft(initial) }, [initial])

  const save = useMutation({
    mutationFn: () =>
      api.upsertEdit(translationId, {
        page_index: bubble.page_index,
        bubble_idx: bubble.bubble_idx,
        edited_text: draft,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['translation', translationId, 'bubbles'] })
      setEditing(false)
    },
  })

  const restore = useMutation({
    mutationFn: () =>
      api.deleteEdit(translationId, bubble.page_index, bubble.bubble_idx),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['translation', translationId, 'bubbles'] })
      setDraft(bubble.draft_text)
      setEditing(false)
    },
  })

  const isEdited = bubble.edited_text != null
  const dirty    = draft !== initial

  return (
    <li className="px-3 py-2.5">
      {/* Source row — always visible, compact */}
      <p className="text-[11px] text-text-subtle leading-relaxed">
        <span className="uppercase font-semibold mr-2">gốc</span>
        <span className="text-text-muted">{bubble.source_text}</span>
      </p>

      {/* Translation row — editable inline */}
      <div className="mt-1 flex items-start gap-2">
        <span className="text-[11px] text-text-subtle uppercase font-semibold mt-1 w-6 shrink-0">
          {bubble.kind === 'sfx'  ? 'sfx' :
           bubble.kind === 'skip' ? 'bỏ' :
                                    'dịch'}
        </span>

        {!editing ? (
          <button
            type="button"
            onClick={() => setEditing(true)}
            className={cn(
              'flex-1 min-w-0 text-left text-[13px] text-text leading-relaxed',
              'rounded-sm px-2 py-1 -mx-2 cursor-text',
              'hover:bg-surface-2 transition-colors',
              isEdited && 'text-amber-400',
            )}
          >
            {draft || <span className="text-text-subtle italic">(trống)</span>}
          </button>
        ) : (
          <textarea
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => {
              if (dirty) save.mutate()
              else setEditing(false)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setDraft(initial)
                setEditing(false)
              } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault()
                if (dirty) save.mutate()
              }
            }}
            rows={Math.max(1, Math.ceil(draft.length / 60))}
            className={cn(
              'flex-1 min-w-0 text-[13px] text-text leading-relaxed',
              'rounded-sm px-2 py-1 -mx-2 bg-surface-2 outline-none',
              'border border-accent/40 focus:border-accent',
              'resize-none',
            )}
          />
        )}

        {isEdited && !editing && (
          <button
            type="button"
            onClick={() => restore.mutate()}
            title="Khôi phục bản LLM"
            className="size-6 rounded-sm flex items-center justify-center text-text-subtle hover:bg-surface-2 hover:text-text shrink-0"
          >
            <RotateCcw size={11} />
          </button>
        )}
      </div>

      {editing && (
        <p className="mt-1 text-[10px] text-text-subtle">
          Esc huỷ · ⌘↵ lưu · blur để lưu
        </p>
      )}
    </li>
  )
}

// ── Route ────────────────────────────────────────────────────────────

export const Route = createFileRoute('/translate/$translationId')({
  validateSearch: (search: Record<string, unknown>): SearchParams => ({
    page: typeof search.page === 'number' ? search.page : undefined,
  }),
  component: TranslationEditorPage,
})
