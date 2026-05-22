// Work-context hooks — direct KV access via /api/works/:id/context.
//
//   useWorkContext(workId)  — server snapshot, returns null if KV miss.
//   useUpdateContext()      — mutation, optimistic-concurrency via version.
//   useDeleteContext()      — mutation, wipes KV blob.
//
// The pipeline auto-merges context on job finalize, so most users never
// need to mutate explicitly — they just submit jobs with work_id and the
// context grows server-side. This is the manual-edit surface (Settings).

import { useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { emptyWorkContext, type WorkContext } from '@shared/db/work-context'

export type { WorkContext }

export interface ContextSnapshot {
  context: WorkContext
  version: number
}

/** Returns the latest server context. `null` means KV miss
 *  (no context exists yet — empty Work). */
export function useWorkContext(workId: string | null | undefined) {
  return useQuery<ContextSnapshot | null, Error>({
    queryKey: workId ? qk.context.byWork(workId) : ['context', 'invalid'],
    queryFn:  () => api.contextGet(workId!),
    enabled:  !!workId,
    staleTime: 30_000,
  })
}

/** Sugar: returns the WorkContext object (always non-null), defaulting
 *  to empty when the server has none yet. */
export function useWorkContextOrEmpty(
  workId: string | null | undefined,
  source_lang: string,
  target_lang: string,
): WorkContext {
  const q = useWorkContext(workId)
  return q.data?.context ?? emptyWorkContext(source_lang, target_lang)
}

export interface UpdateContextArgs {
  work_id: string
  next:    WorkContext
  /** Pass the version from the last GET to enable optimistic concurrency;
   *  pass `null` to skip the check (last-writer-wins). */
  base_version: number | null
}

export function useUpdateContext() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ work_id, next, base_version }: UpdateContextArgs) =>
      api.contextPut(work_id, next, base_version),
    onSuccess: (_result, args) => {
      qc.invalidateQueries({ queryKey: qk.context.byWork(args.work_id) })
    },
  })
}

export function useDeleteContext() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (work_id: string) => api.contextDelete(work_id),
    onSuccess: (_v, work_id) => {
      qc.removeQueries({ queryKey: qk.context.byWork(work_id) })
    },
  })
}

/** Manual refresh — useful right after a job finishes, before the
 *  TanStack staleTime expires. */
export function useRefetchContext() {
  const qc = useQueryClient()
  return useCallback((work_id: string) => {
    qc.invalidateQueries({ queryKey: qk.context.byWork(work_id) })
  }, [qc])
}
