// Hooks for the "Nguồn" tab on the Work hub — listing current
// members + casting split votes + the owner force-unlink undo.
//
// One module owns the API + cache invalidation for every mutation
// that changes which materials belong to a Work. The
// `LinkSuggestionPanel` (merge side) still has its own mutation
// inline; if/when we consolidate, the merge mutations will move
// here too. For now this module owns the split side only — keeps
// the diff small and the dual-track concern (members panel vs
// suggestions panel) cleanly separated.

import { useCallback } from 'react'
import {
  useMutation, useQuery, useQueryClient,
} from '@tanstack/react-query'

import { api, type ApiSplitVoteResult, type ApiWorkMember } from '@shared/api/api'
import { qk } from '@shared/api/keys'


/** List materials currently attached to this Work, with viewer
 *  split-vote state + owner-undo hint folded in. Refetched after
 *  every split / unlink mutation so the panel always reflects
 *  truth. */
export function useWorkMembers(workId: number) {
  return useQuery({
    queryKey:  qk.work.members(workId),
    queryFn:   () => api.listWorkMembers(workId),
    staleTime: 30_000,
  })
}


/** Cache invalidation shared by every mutation in this module.
 *  Touching split state always re-reads the members list AND the
 *  Work payload (the latter carries materials too, which the hub
 *  hero / chapter list read). */
function useInvalidate(workId: number) {
  const qc = useQueryClient()
  return useCallback(() => {
    void qc.invalidateQueries({ queryKey: qk.work.members(workId) })
    void qc.invalidateQueries({ queryKey: qk.work.byId(workId) })
    void qc.invalidateQueries({ queryKey: qk.work.linkSuggest(workId) })
    void qc.invalidateQueries({ queryKey: qk.library.all() })
  }, [qc, workId])
}


/** ±1 vote: "tách nguồn này khỏi truyện". When the score crosses
 *  the threshold and the work has ≥2 members, the material moves
 *  to a fresh isolated work — the caller receives the new work id
 *  and can navigate / toast accordingly. */
export function useCastSplitVote(workId: number) {
  const invalidate = useInvalidate(workId)
  return useMutation<
    ApiSplitVoteResult,
    Error,
    { material_id: number; vote: 1 | -1 }
  >({
    mutationFn: (body) => api.castWorkSplitVote(workId, body),
    onSuccess:  () => invalidate(),
  })
}


/** Owner-only undo for a recent force_link, instant — no community
 *  vote required, no threshold. Server gates the undo window (10
 *  minutes); after that this endpoint returns 403 and the caller
 *  has to go through `castSplitVote` instead. */
export function useForceUnlink(workId: number) {
  const invalidate = useInvalidate(workId)
  return useMutation<
    ApiSplitVoteResult,
    Error,
    { material_id: number }
  >({
    mutationFn: (body) => api.forceWorkUnlink(workId, body),
    onSuccess:  () => invalidate(),
  })
}


/** Convenience predicate: is `member` still inside the owner undo
 *  window right now? Re-evaluated by callers on a 1-second tick so
 *  the affordance disappears when the window closes without waiting
 *  for a refetch. */
export function isUndoWindowOpen(
  member: Pick<ApiWorkMember, 'force_link_undo_expires_at'>,
  now: number = Date.now(),
): boolean {
  const iso = member.force_link_undo_expires_at
  if (!iso) return false
  const expires = Date.parse(iso)
  return Number.isFinite(expires) && expires > now
}
