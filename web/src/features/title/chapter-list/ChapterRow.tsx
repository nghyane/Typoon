// ChapterRow — pure presentation. Surface owns:
//   • action callbacks (`actions`)
//   • spawn progress state (`spawn`, null when surface doesn't spawn)
//   • selection (`selection`, null when surface is read-only)
//
// Reader navigation is internal: row click routes to /read or /raw
// based on what's available. Doing it here keeps every chapter-list
// surface consistent (same clickability rules everywhere).

import { useNavigate } from '@tanstack/react-router'
import { AlertCircle, Loader2 } from 'lucide-react'
import { cn } from '@shared/lib/cn'
import {
  inFlight, lastError, preferredReadable, stripChapterPrefix,
  chapterStatus,
  type HubChapter, type HubVersion,
} from '../mergeChapters'
import { Checkbox } from './Checkbox'
import { ChapterRowActions } from './ChapterRowActions'
import { TimeCell } from './TimeCell'
import type {
  RowActions, RowContext, SelectionState,
} from './types'
import type { SpawnProgress } from '../useSpawnChapter'

interface Props {
  chapter:       HubChapter
  targetLang:    string | null
  materialTitle: string
  actions:       RowActions
  /** Selection state. null on surfaces without bulk actions
   *  (public-read /material/$id). */
  selection:     SelectionState | null
  /** Live spawn progress for this row, when the surface drives spawn.
   *  null when no spawn affordance — the row renders translation /
   *  raw states only. */
  spawn:         SpawnProgress | null
  spawnReset:    () => void
}

export function ChapterRow({
  chapter, targetLang, materialTitle,
  actions, selection, spawn, spawnReset,
}: Props) {
  const nav = useNavigate()

  const status   = chapterStatus(chapter, targetLang)
  const readable = preferredReadable(chapter, targetLang)
  const running  = inFlight(chapter, targetLang)
  const errored  = lastError(chapter, targetLang)
  const label    = stripChapterPrefix(chapter.label, chapter.number)

  const spawning = !!spawn
    && spawn.phase !== 'idle'
    && spawn.phase !== 'done'
    && spawn.phase !== 'error'

  // Resolve row context — fed to action callbacks so the surface can
  // act without re-reading the chapter shape.
  const ctx: RowContext = resolveRowContext(chapter, readable)
  const checked       = selection?.has(chapter.number) ?? false
  const anySelected   = (selection?.size ?? 0) > 0

  const stripeColor =
    status === 'translated' ? 'var(--color-accent)'
  : status === 'running'    ? 'var(--color-info)'
  : status === 'error'      ? 'var(--color-error)'
                            : 'transparent'

  const rowClickable =
    !spawning && (
      ctx.doneTranslation !== null
      || status === 'running'
      || (status === 'raw' && !!ctx.anyRaw)
    )

  const onRowClick = () => {
    if (!rowClickable) return
    // Let surface override if it wired one.
    if (actions.onRowClick) {
      actions.onRowClick(ctx)
      return
    }
    if (ctx.doneTranslation?.translationId != null) {
      nav({
        to: '/read/$translationId',
        params: { translationId: String(ctx.doneTranslation.translationId) },
      })
      return
    }
    if (status === 'running' && running?.translationId != null) {
      nav({
        to: '/read/$translationId',
        params: { translationId: String(running.translationId) },
      })
      return
    }
    if (status === 'raw' && ctx.anyRaw?.upstreamUrl && ctx.anyRaw.sourceId) {
      nav({
        to: '/raw',
        search: {
          source:     ctx.anyRaw.sourceId,
          url:        ctx.anyRaw.upstreamUrl,
          title:      materialTitle,
          number:     chapter.number,
          label:      chapter.label ?? undefined,
          materialId: ctx.anyRaw.materialId,
          numberNorm: ctx.anyRaw.numberNorm ?? undefined,
        },
      })
    }
  }

  return (
    <tr
      onClick={onRowClick}
      className={cn(
        'group border-b border-border-soft last:border-0 transition-colors',
        checked ? 'bg-row-active' : 'hover:bg-hover',
        rowClickable && 'cursor-pointer',
      )}
    >
      {selection && (
        <td
          className="pl-3 pr-1 py-3 w-8"
          style={{ boxShadow: `inset 2px 0 0 0 ${stripeColor}` }}
          onClick={(e) => e.stopPropagation()}
        >
          <div className={cn(
            'transition-opacity',
            checked || anySelected
              ? 'opacity-100'
              : 'opacity-0 group-hover:opacity-100',
          )}>
            <Checkbox
              checked={checked}
              onClick={() => selection.toggle(chapter.number)}
              ariaLabel="Chọn chương"
            />
          </div>
        </td>
      )}
      {!selection && (
        <td
          className="pl-3 pr-1 py-3 w-2"
          style={{ boxShadow: `inset 2px 0 0 0 ${stripeColor}` }}
        />
      )}

      <td className="pr-3 py-3 whitespace-nowrap tabular text-right font-semibold text-text">
        {chapter.number}
      </td>

      <td className="px-3 py-3 min-w-0 w-full">
        {label
          ? <div className="text-sm text-text truncate">{label}</div>
          : <div className="text-sm text-text-subtle">—</div>
        }
        <RowMeta
          spawn={spawn}
          spawnReset={spawnReset}
          status={status}
          running={running}
          errored={errored}
        />
      </td>

      <td className="px-3 py-3 hidden sm:table-cell whitespace-nowrap text-right w-px">
        <TimeCell iso={chapter.updatedAt} />
      </td>

      <td
        className="pl-2 pr-3 py-3 whitespace-nowrap text-right"
        onClick={(e) => e.stopPropagation()}
      >
        <ChapterRowActions
          status={status}
          spawning={spawning}
          ctx={ctx}
          actions={actions}
        />
      </td>
    </tr>
  )
}


// ── Sub-line under the chapter label ─────────────────────────────────

function RowMeta({
  spawn, spawnReset, status, running, errored,
}: {
  spawn:      SpawnProgress | null
  spawnReset: () => void
  status:     ReturnType<typeof chapterStatus>
  running:    HubVersion | null
  errored:    HubVersion | null
}) {
  if (spawn?.phase === 'error') {
    return (
      <div className="mt-0.5 inline-flex items-center gap-1 text-xs text-error-text">
        <AlertCircle size={12} className="shrink-0" />
        <span className="truncate">{spawn.error}</span>
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); spawnReset() }}
          className="underline ml-1 cursor-pointer"
        >
          Đóng
        </button>
      </div>
    )
  }
  if (spawn && spawn.phase !== 'idle' && spawn.phase !== 'done') {
    return (
      <div className="mt-0.5 inline-flex items-center gap-1 text-xs text-info-text">
        <Loader2 size={12} className="animate-spin shrink-0" />
        <SpawnLabel spawn={spawn} />
      </div>
    )
  }
  if (status === 'running') {
    return (
      <div className="mt-0.5 inline-flex items-center gap-1 text-xs text-info-text">
        <Loader2 size={12} className="animate-spin shrink-0" />
        {running?.creatorName ? `@${running.creatorName} đang dịch` : 'Đang dịch'}
      </div>
    )
  }
  if (status === 'error') {
    return (
      <div className="mt-0.5 inline-flex items-center gap-1 text-xs text-error-text">
        <AlertCircle size={12} className="shrink-0" />
        {errored?.creatorName ? `@${errored.creatorName} · lỗi` : 'Lỗi dịch'}
      </div>
    )
  }
  return null
}


function SpawnLabel({ spawn }: { spawn: SpawnProgress }) {
  switch (spawn.phase) {
    case 'fetching':    return <>Đang lấy danh sách trang…</>
    case 'downloading': return <>{spawn.current}/{spawn.total} trang</>
    case 'packing':     return <>Đang nén…</>
    case 'uploading':   return <>Đang tải lên {spawn.pct}%</>
    case 'spawning':    return <>Đang khởi động dịch…</>
    default:            return null
  }
}


// ── Row-context resolver ──────────────────────────────────────────

function resolveRowContext(
  chapter:  HubChapter,
  readable: HubVersion | null,
): RowContext {
  const spawnableRaw = chapter.versions.find(
    (v) => v.kind === 'raw' && !!v.upstreamUrl && !!v.sourceId,
  ) ?? null
  return {
    spawnableRaw,
    anyRaw: spawnableRaw, // same condition today; kept distinct for future divergence
    doneTranslation: readable?.kind === 'translation' ? readable : null,
  }
}
