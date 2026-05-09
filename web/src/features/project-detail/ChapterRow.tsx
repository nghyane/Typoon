import { Check, Eye, Play, RefreshCw, AlertCircle, Trash2 } from 'lucide-react'
import { Link } from '@tanstack/react-router'
import type { ApiChapter } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { Spinner } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import { STATE_LABEL, STATE_TONE, stageLabelFor, chapterPct } from './chapter'
import type { useChapterMutations } from './mutations'

const PROGRESS_BAR_BY_TONE: Record<string, string> = {
  success: 'bg-success',
  info:    'bg-info',
  warning: 'bg-warning',
  error:   'bg-error',
  neutral: 'bg-text-subtle',
}

const DOT_BY_TONE: Record<string, string> = {
  success: 'bg-success',
  info:    'bg-info',
  warning: 'bg-warning',
  error:   'bg-error',
  neutral: 'bg-text-subtle',
}

interface Props {
  ch:        ApiChapter
  checked:   boolean
  onToggle:  () => void
  isOwner:   boolean
  projectId: number
  mutations: ReturnType<typeof useChapterMutations>
}

export function ChapterRow({ ch, checked, onToggle, isOwner, projectId, mutations }: Props) {
  const tone   = STATE_TONE[ch.state]
  const label  = STATE_LABEL[ch.state]
  // Stage-aware label: distinguishes "Chờ quét" (pending) from
  // "Đang quét" (running). Falls back to plain state label when stage
  // is unknown (idle/done/error).
  const sub      = ch.stage ? stageLabelFor(ch.state, ch.stage) : label
  const running  = ch.state === 'running'
  const pending  = ch.state === 'pending'
  const inFlight = running || pending
  const pct      = chapterPct(ch)
  const showBar  = running   // only running has real progress data

  const counter = ch.state === 'done'
    ? `${ch.page_count} / ${ch.page_count}`
    : running && ch.progress
    ? `${ch.progress.page_index} / ${ch.progress.page_total}`
    : `${ch.page_count} trang`

  const redoPending   = mutations.redo.isPending   && mutations.redo.variables   === ch.chapter_id
  const startPending  = mutations.start.isPending  && mutations.start.variables  === ch.chapter_id
  const removePending = mutations.remove.isPending && mutations.remove.variables === ch.chapter_id

  return (
    <tr
      className={cn(
        'group transition-colors',
        'border-b border-border-soft last:border-0',
        checked ? 'bg-row-active' : 'hover:bg-hover',
      )}
      style={checked ? { boxShadow: 'inset 2px 0 0 0 var(--color-accent)' } : undefined}
    >
      <td className="px-3 py-3 w-10">
        <button
          onClick={onToggle}
          aria-label={checked ? 'Bỏ chọn' : 'Chọn'}
          className={cn(
            'size-4 rounded-xs border flex items-center justify-center cursor-pointer transition-colors',
            checked
              ? 'bg-accent border-accent text-accent-fg'
              : 'border-text-subtle hover:border-text-muted',
          )}
        >
          {checked && <Check size={9} strokeWidth={3} />}
        </button>
      </td>

      <td className="px-3 py-3 min-w-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="font-semibold text-text tabular shrink-0">
            Ch.{ch.number}
          </span>
          {ch.title && (
            <span className="text-sm text-text-muted truncate">{ch.title}</span>
          )}
        </div>
        {ch.state === 'error' && ch.error && (
          <div className="mt-1 inline-flex items-center gap-1.5 text-xs text-text-subtle truncate max-w-md">
            <AlertCircle size={11} className="shrink-0 text-error-text" />
            <span className="truncate" title={ch.error}>{ch.error}</span>
          </div>
        )}
      </td>

      <td className="px-3 py-3 w-64">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-xs">
            {running ? (
              <Spinner size={10} className="text-info-text" />
            ) : (
              <span className={cn('size-1.5 rounded-full shrink-0', DOT_BY_TONE[tone])} />
            )}
            <span className={cn(
              'whitespace-nowrap',
              running              ? 'text-info-text'
              : ch.state === 'error' ? 'text-error-text'
              : pending              ? 'text-text-muted'
              : 'text-text-muted',
            )}>{sub}</span>
            <span className="ml-auto text-[11px] tabular text-text-subtle">{counter}</span>
          </div>
          {showBar && (
            <div className="h-[3px] rounded-full bg-surface-2 overflow-hidden">
              <div
                className={cn('h-full rounded-full transition-[width] duration-300', PROGRESS_BAR_BY_TONE[tone])}
                style={{ width: `${pct}%` }}
              />
            </div>
          )}
        </div>
      </td>

      <td className="px-3 py-3 text-xs text-text-subtle whitespace-nowrap w-24 tabular" title={ch.updated_at ?? ''}>
        {timeAgo(ch.updated_at)}
      </td>

      <td className="px-3 py-3 w-32">
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity justify-end">
          {ch.state === 'done' ? (
            <Link
              to="/projects/$projectId/chapters/$chapterId"
              params={{ projectId: String(projectId), chapterId: String(ch.chapter_id) }}
              title="Xem render"
            >
              <Button variant="ghost" size="sm" icon>
                <Eye size={14} />
              </Button>
            </Link>
          ) : (
            <Button variant="ghost" size="sm" icon disabled title="Chưa render xong">
              <Eye size={14} />
            </Button>
          )}
          {isOwner && (
            <>
              {/* idle → "Bắt đầu dịch"; non-idle → "Chạy lại" (redo
                  resets derived data first). One slot, one icon, the
                  semantics swap with the chapter state. */}
              {ch.state === 'idle' ? (
                <Button
                  variant="ghost"
                  size="sm"
                  icon
                  title="Bắt đầu dịch"
                  disabled={startPending}
                  onClick={() => mutations.start.mutate(ch.chapter_id)}
                >
                  <Play size={14} className={startPending ? 'animate-pulse' : ''} />
                </Button>
              ) : (
                <Button
                  variant="ghost"
                  size="sm"
                  icon
                  title="Chạy lại"
                  disabled={redoPending || inFlight}
                  onClick={() => mutations.redo.mutate(ch.chapter_id)}
                >
                  <RefreshCw size={14} className={redoPending ? 'animate-spin' : ''} />
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                icon
                className="hover:text-error-text"
                title="Xoá"
                disabled={removePending || inFlight}
                onClick={() => {
                  if (confirm(`Xoá Ch.${ch.number}?`))
                    mutations.remove.mutate(ch.chapter_id)
                }}
              >
                <Trash2 size={14} />
              </Button>
            </>
          )}
        </div>
      </td>
    </tr>
  )
}
