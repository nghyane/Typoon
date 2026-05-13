// LinkSuggestionPanel — community vote UI for cross-source linking.
//
// Shows materials *outside* this Work that the community has voted
// to merge in (score > 0). Each row has:
//   • a thumbnail + source label
//   • the current vote score
//   • viewer's action: Đồng ý (+1) or Không phải (−1).  When the
//     viewer has already voted, the row reflects it.
//
// When a vote crosses the server-side threshold (3 distinct users)
// AND the two Works don't carry conflicting cross_refs, the server
// merges them inline. The mutation surfaces `merged: true` plus a
// `canonical_work_id` — the SPA navigates there so the user lands
// on the surviving Work id.
//
// Conflict (`blocked_reason: 'cross_refs_conflict'`) is rendered
// as a passive notice; the vote was still recorded for moderation.

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Check, X, Link2 } from 'lucide-react'

import {
  api, type ApiLinkSuggestion, type ApiLinkVoteResult, type ApiMaterial,
} from '@shared/api/api'
import { Cover } from '@shared/ui/Cover'
import { Spinner } from '@shared/ui/primitives'
import { cn } from '@shared/lib/cn'
import { qk } from '@shared/api/keys'
import { useSources } from '@features/browse/sources'
import { confirm } from '@shared/ui/Confirm'
import { toast } from '@shared/ui/Toaster'

import { LinkSearchExpander } from './LinkSearchExpander'
import type { SearchHit } from '@features/library/addManga/fanoutSearch'


interface Props {
  workId:       number
  /** Materials currently on this Work. Used by the search expander
   *  to filter self-link candidates and badge "already in library"
   *  rows. */
  ownMaterials: ApiMaterial[]
}


export function LinkSuggestionPanel({ workId, ownMaterials }: Props) {
  const qc = useQueryClient()
  const nav = useNavigate()

  const sugQ = useQuery({
    queryKey: qk.work.linkSuggest(workId),
    queryFn:  () => api.listWorkLinkSuggestions(workId),
    staleTime: 30_000,
  })
  const list = sugQ.data ?? []

  const importAndLink = useMutation({
    mutationFn: async (hit: SearchHit) => {
      // Two steps: import the candidate as a new material (the
      // linker assigns it to whichever Work shares cross_refs, or a
      // fresh isolated Work otherwise), then cast a +1 vote linking
      // it to this Work. Server merges when threshold hits.
      const m = await api.importMaterial({
        source:       hit.source.manifest.id,
        upstream_ref: hit.manga.url,
        title:        hit.manga.title,
        cover_url:    hit.manga.cover ?? null,
      })
      const own = ownMaterials[0]
      if (!own) throw new Error('Work has no own material to vote with.')
      const res = await api.castWorkLinkVote(workId, {
        target_material_id: m.id,
        own_material_id:    own.id,
        vote:               1,
      })
      return { res, importedTitle: m.title }
    },
    onSuccess: ({ res, importedTitle }) => {
      qc.invalidateQueries({ queryKey: qk.work.byId(workId) })
      qc.invalidateQueries({ queryKey: qk.work.linkSuggest(workId) })
      qc.invalidateQueries({ queryKey: qk.library.all() })
      if (res.merged && res.canonical_work_id != null
          && res.canonical_work_id !== workId) {
        toast.success(`Đã gộp với "${importedTitle}"`)
        nav({
          to:     '/w/$workId',
          params: { workId: String(res.canonical_work_id) },
        })
        return
      }
      toast.success(`Đã thêm "${importedTitle}" và ghi nhận liên kết`)
    },
    onError: (e: Error) => {
      toast.error(`Liên kết thất bại: ${e.message}`)
    },
  })

  const handlePick = async (hit: SearchHit) => {
    const proceed = await confirm({
      title:       'Liên kết manga?',
      description: `Thêm "${hit.manga.title}" từ ${hit.source.manifest.name} và đánh dấu là cùng truyện?`,
      confirmText: 'Liên kết',
    })
    if (!proceed) return
    importAndLink.mutate(hit)
  }

  // Hide the whole panel only when there's nothing to suggest AND
  // the search affordance can't be useful (e.g. work has no own
  // material to vote with — extremely rare; happens mid-redirect).
  if (sugQ.isPending) {
    return (
      <Section>
        <div className="py-3 flex justify-center">
          <Spinner size={16} />
        </div>
      </Section>
    )
  }

  const hasSearch = ownMaterials.length > 0
  if (list.length === 0 && !hasSearch) return null

  return (
    <Section>
      {list.length > 0 && (
        <ul className="space-y-1 mb-3">
          {list.map((s) => (
            <SuggestionRow key={s.candidate_material_id} suggestion={s} workId={workId} />
          ))}
        </ul>
      )}
      {hasSearch && (
        <LinkSearchExpander
          ownMaterials={ownMaterials}
          onPick={handlePick}
        />
      )}
    </Section>
  )
}


function Section({ children }: { children: React.ReactNode }) {
  return (
    <section className="px-4 sm:px-6 py-3">
      <h2 className="text-xs uppercase tracking-wide text-text-subtle flex items-center gap-1.5 mb-2">
        <Link2 size={12} />
        Có thể là cùng manga
      </h2>
      {children}
    </section>
  )
}


function SuggestionRow({
  suggestion, workId,
}: {
  suggestion: ApiLinkSuggestion
  workId:     number
}) {
  const qc = useQueryClient()
  const nav = useNavigate()
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
        // The current Work dissolved into a sibling — jump there.
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

  const voted = suggestion.viewer_vote
  const busy  = vote.isPending
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
        <div className="text-xs text-text-subtle truncate">
          {sourceLabel} · {describeSignal(suggestion)}
        </div>
        {blocked && (
          <div className="text-[11px] text-amber-400 mt-0.5">
            Không thể gộp tự động — hai nguồn khai báo định danh khác nhau.
          </div>
        )}
      </div>

      {voted === 1 ? (
        <span className="text-xs text-accent inline-flex items-center gap-1">
          <Check size={12} /> Đã đồng ý
        </span>
      ) : voted === -1 ? (
        <span className="text-xs text-text-subtle inline-flex items-center gap-1">
          <X size={12} /> Đã từ chối
        </span>
      ) : (
        <div className="flex items-center gap-1">
          <button
            type="button"
            disabled={busy}
            onClick={() => vote.mutate(1)}
            className={cn(
              'h-7 px-2 rounded-sm text-xs cursor-pointer inline-flex items-center gap-1',
              'bg-accent/15 text-accent hover:bg-accent/25',
              busy && 'opacity-60 cursor-wait',
            )}
            title="Đồng ý: 2 nguồn này cùng 1 manga"
          >
            <Check size={12} />
            Đúng
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => vote.mutate(-1)}
            className={cn(
              'h-7 px-2 rounded-sm text-xs cursor-pointer inline-flex items-center gap-1',
              'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
              busy && 'opacity-60 cursor-wait',
            )}
            title="Không, manga khác"
          >
            <X size={12} />
            Khác
          </button>
        </div>
      )}
    </li>
  )
}


/** Render the "why this row showed up" meta line. Voted candidates
 *  surface the community score; ranker candidates surface the signal
 *  the ranker latched onto (matching Japanese title, alt-title
 *  overlap, or fuzzy similarity) so the user knows what to verify
 *  before voting. */
function describeSignal(s: ApiLinkSuggestion): string {
  if (s.kind === 'voted') {
    return `${s.total_votes} người vote · score ${s.score}`
  }
  const conf = s.confidence != null
    ? `${Math.round(s.confidence * 100)}%`
    : ''
  switch (s.reason) {
    case 'title_native_exact':
      return `Tên gốc trùng${conf ? ` · ${conf}` : ''}`
    case 'title_alt_overlap':
      return `Tên gọi khác trùng${conf ? ` · ${conf}` : ''}`
    case 'title_trgm':
      return `Tên giống${conf ? ` ${conf}` : ''}`
    default:
      return conf || 'Gợi ý'
  }
}
