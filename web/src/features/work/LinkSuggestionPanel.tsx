// LinkSuggestionPanel — cross-source linking surface on the Work page.
//
// Lives inside the Work page section stack alongside WorkChapterList:
// flat section (no card chrome), `px-4 sm:px-6` route padding, design
// tokens via shared primitives (`Button`, `Badge`, `Tag`, `Spinner`,
// `input`, `card`). Two write paths share the same surface:
//
//   1. Suggestion rows — auto-ranker + community votes. ±1 buttons
//      go through `link-vote`; merge fires only after the community
//      crosses `LINK_MERGE_THRESHOLD` server-side.
//
//   2. Manual search — opens a modal (`LinkSearchModal`) mirroring
//      `AddMangaModal`'s shell. The picked hit funnels through
//      `importMaterialFromHit` + `force-link`, which bypasses the
//      threshold for the affirmative-intent case. cross_refs
//      conflicts still fall back to "vote recorded, merge refused".
//
// Visual hierarchy follows the existing Work page: heading uses the
// section header pattern (`text-xs uppercase tracking-wide`), rows
// reuse the chapter-row hover treatment (`bg-surface-2/40 hover:…`),
// the manual-link trigger lives in the header action slot so the
// section keeps its content-first layout when no one opens it.
//
// Threshold display: `LINK_MERGE_THRESHOLD` is mirrored from
// `typoon/api/routes/work.py`. Server is the source of truth — the
// constant here only drives the "còn cần N phiếu" hint. Mismatches
// degrade gracefully.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Check, X, Link2, Plus, Undo2 } from 'lucide-react'
import { useMemo, useState } from 'react'

import {
  api, type ApiLinkSuggestion, type ApiLinkVoteResult, type ApiMaterial,
} from '@shared/api/api'
import { Cover } from '@shared/ui/Cover'
import { Badge, Spinner, Tag } from '@shared/ui/primitives'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { qk } from '@shared/api/keys'
import { useSources } from '@features/browse/sources'

import { LinkSearchModal } from './LinkSearchModal'
import { resolveWorkTitle } from './title'


/** Mirror of `LINK_MERGE_THRESHOLD` in `typoon/api/routes/work.py`.
 *  Display-only — server still gates the actual merge. */
const LINK_MERGE_THRESHOLD = 2


interface Props {
  workId:       number
  /** Materials currently on this Work. Used by the search modal to
   *  filter self-link candidates and to anchor the vote pair. */
  ownMaterials: ApiMaterial[]
  /** Viewer's reading language — drives the modal subtitle so the
   *  user sees the same title they recognise on the Work page. */
  targetLang:   string | null
}


export function LinkSuggestionPanel({
  workId, ownMaterials, targetLang,
}: Props) {
  const [searchOpen, setSearchOpen] = useState(false)

  const sugQ = useQuery({
    queryKey: qk.work.linkSuggest(workId),
    queryFn:  () => api.listWorkLinkSuggestions(workId),
    staleTime: 30_000,
  })

  const workTitle = useMemo(
    () => resolveWorkTitle(ownMaterials, targetLang).title,
    [ownMaterials, targetLang],
  )

  const list      = sugQ.data ?? []
  const hasSearch = ownMaterials.length > 0

  if (!sugQ.isPending && list.length === 0 && !hasSearch) return null

  return (
    <section className="px-4 sm:px-6 py-3 space-y-3">
      <SectionHeader
        count={sugQ.isPending ? null : list.length}
        action={hasSearch && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSearchOpen(true)}
          >
            <Plus size={12} />
            Liên kết nguồn khác
          </Button>
        )}
      />

      {sugQ.isPending ? (
        <div className="py-4 flex justify-center">
          <Spinner size={16} />
        </div>
      ) : list.length > 0 ? (
        <ul className="space-y-1">
          {list.map((s) => (
            <SuggestionRow
              key={s.candidate_material_id}
              suggestion={s}
              workId={workId}
            />
          ))}
        </ul>
      ) : (
        <p className="text-sm text-text-subtle">
          Chưa có đề xuất nào. Nếu bạn biết manga này ở nguồn khác, bấm
          “Liên kết nguồn khác” ở góc trên.
        </p>
      )}

      {hasSearch && (
        <LinkSearchModal
          open={searchOpen}
          onClose={() => setSearchOpen(false)}
          workId={workId}
          workTitle={workTitle}
          ownMaterials={ownMaterials}
        />
      )}
    </section>
  )
}


// ── Section header ─────────────────────────────────────────────


function SectionHeader({
  count, action,
}: {
  count:   number | null
  action?: React.ReactNode
}) {
  return (
    <div className="flex items-center gap-2">
      <h2 className="text-xs uppercase tracking-wide text-text-subtle inline-flex items-center gap-1.5">
        <Link2 size={12} />
        Đề xuất gộp
      </h2>
      {count != null && count > 0 && (
        <Tag tone="neutral" size="sm">{count}</Tag>
      )}
      {action && <div className="ml-auto">{action}</div>}
    </div>
  )
}


// ── Suggestion row ─────────────────────────────────────────────


function SuggestionRow({
  suggestion, workId,
}: {
  suggestion: ApiLinkSuggestion
  workId:     number
}) {
  const qc        = useQueryClient()
  const nav       = useNavigate()
  const installed = useSources((s) => s.sources)

  const sourceLabel = suggestion.candidate_source
    ? (installed[suggestion.candidate_source]?.manifest.name
       ?? suggestion.candidate_source)
    : 'Tải lên'

  const vote = useMutation({
    mutationFn: (v: 1 | -1) =>
      api.castWorkLinkVote(workId, {
        target_material_id: suggestion.candidate_material_id,
        own_material_id:    suggestion.own_material_id,
        vote:               v,
      }),
    onSuccess: (res: ApiLinkVoteResult) => {
      if (res.merged && res.canonical_work_id != null
          && res.canonical_work_id !== workId) {
        nav({
          to:     '/w/$workId',
          params: { workId: String(res.canonical_work_id) },
        })
        return
      }
      qc.invalidateQueries({ queryKey: qk.work.byId(workId) })
      qc.invalidateQueries({ queryKey: qk.work.linkSuggest(workId) })
    },
  })

  const voted   = suggestion.viewer_vote
  const busy    = vote.isPending
  const blocked = vote.data?.blocked_reason === 'cross_refs_conflict'

  return (
    <li
      className={cn(
        'flex items-center gap-3 p-2 rounded-sm transition-colors',
        'bg-surface-2/40 hover:bg-surface-2/70',
      )}
    >
      <div className="w-10 h-14 shrink-0 rounded-sm overflow-hidden">
        <Cover
          src={suggestion.candidate_cover}
          title={suggestion.candidate_title}
          className="w-full h-full"
          fontSize="text-xs"
        />
      </div>

      <div className="flex-1 min-w-0">
        <div className="text-sm text-text truncate">
          {suggestion.candidate_title}
        </div>
        <div className="flex items-center gap-1.5 mt-0.5 min-w-0">
          <span className="text-xs text-text-subtle truncate">
            {sourceLabel}
          </span>
          <SignalTag suggestion={suggestion} />
        </div>
        {blocked && (
          <p className="text-xs text-warning-text mt-1">
            Không thể gộp tự động — hai nguồn khai báo định danh khác nhau.
          </p>
        )}
      </div>

      <VoteAction
        voted={voted}
        busy={busy}
        suggestion={suggestion}
        onVote={(v) => vote.mutate(v)}
      />
    </li>
  )
}


// ── Vote action ────────────────────────────────────────────────


/** Undo flips the vote (API has no neutral state). */
function VoteAction({
  voted, busy, suggestion, onVote,
}: {
  voted:      number | null
  busy:       boolean
  suggestion: ApiLinkSuggestion
  onVote:     (v: 1 | -1) => void
}) {
  if (voted === 1) {
    const score = suggestion.kind === 'voted' ? suggestion.score : 1
    const need  = Math.max(0, LINK_MERGE_THRESHOLD - score)
    return (
      <div className="shrink-0 flex flex-col items-end gap-1">
        <Badge tone="success">
          Đã đồng ý
          {need > 0 && (
            <span className="tabular text-text-muted font-normal">
              {score}/{LINK_MERGE_THRESHOLD}
            </span>
          )}
        </Badge>
        <UndoLink busy={busy} onClick={() => onVote(-1)} />
      </div>
    )
  }
  if (voted === -1) {
    return (
      <div className="shrink-0 flex flex-col items-end gap-1">
        <Badge tone="neutral">Đã từ chối</Badge>
        <UndoLink busy={busy} onClick={() => onVote(1)} />
      </div>
    )
  }
  return (
    <div className="shrink-0 flex items-center gap-1">
      <Button
        variant="primary"
        size="sm"
        icon
        disabled={busy}
        onClick={() => onVote(1)}
        title="Đồng ý: 2 nguồn này cùng 1 manga"
        aria-label="Đồng ý"
      >
        <Check size={14} />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        icon
        disabled={busy}
        onClick={() => onVote(-1)}
        title="Không, manga khác"
        aria-label="Từ chối"
      >
        <X size={14} />
      </Button>
    </div>
  )
}


function UndoLink({ busy, onClick }: { busy: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      disabled={busy}
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1 text-xs cursor-pointer',
        'text-text-subtle hover:text-text transition-colors',
        busy && 'opacity-60 cursor-wait',
      )}
    >
      <Undo2 size={11} /> Hoàn tác
    </button>
  )
}



// ── Signal tag ─────────────────────────────────────────────────


function SignalTag({ suggestion: s }: { suggestion: ApiLinkSuggestion }) {
  if (s.kind === 'voted') {
    return (
      <Tag tone="info" size="sm">
        {s.total_votes} phiếu
      </Tag>
    )
  }
  const conf = s.confidence != null
    ? ` · ${Math.round(s.confidence * 100)}%`
    : ''
  switch (s.reason) {
    case 'title_native_exact':
      return <Tag tone="success" size="sm">Tên gốc trùng{conf}</Tag>
    case 'title_alt_overlap':
      return <Tag tone="info"    size="sm">Tên gọi khác{conf}</Tag>
    case 'title_trgm':
      return <Tag tone="outline" size="sm">Tên giống{conf}</Tag>
    default:
      return <Tag tone="outline" size="sm">Gợi ý{conf}</Tag>
  }
}
