// WorkIdentityContext — small surface, infrequent updates.
//
// Holds the work entity, the primary source pick, and the cached
// translation context. Identity rarely changes during a session;
// keeping it in its own context prevents chapter list re-renders from
// invalidating heavy consumers like the hero.

import { createContext, useContext, useMemo, type ReactNode } from 'react'

import { useSessionUser } from '@features/auth/session'
import { resolveReadingLang, FALLBACK_LANG } from '@features/auth/readingLang'
import { useWorkContext } from '@features/library/context'

import { pickPrimarySourceIndex } from '../data/selectors/primarySource'
import { useWorkManifests } from '../data/queries/useWorkManifests'
import type {
  ContextSnapshot, MangaDetail, Work,
} from '../data/types'


export interface WorkIdentity {
  work:          Work
  workId:        string
  primaryIdx:    number
  primaryDetail: MangaDetail | null
  /** All manifest details (same length as work.sources). Some entries
   *  may be undefined while loading or when the source adapter is
   *  disabled. The chapter list consumer needs the full array. */
  manifestDetails: (MangaDetail | undefined)[]
  manifestsLoading: boolean
  workCtx:       ContextSnapshot | null
}


const Ctx = createContext<WorkIdentity | null>(null)


export function useWorkIdentity(): WorkIdentity {
  const v = useContext(Ctx)
  if (!v) throw new Error('useWorkIdentity must be used inside <WorkIdentityProvider>')
  return v
}


interface Props {
  work:     Work
  workId:   string
  children: ReactNode
}


export function WorkIdentityProvider({ work, workId, children }: Props) {
  const sessionUser = useSessionUser()
  const manifests   = useWorkManifests(work.sources)
  const workCtxQ    = useWorkContext(workId)

  const readingLang = resolveReadingLang(
    work.target_lang,
    sessionUser?.preferred_target_lang,
  ) || FALLBACK_LANG

  const primaryIdx = useMemo(
    () => pickPrimarySourceIndex(work.sources, readingLang),
    [work.sources, readingLang],
  )

  const primaryDetail = primaryIdx >= 0
    ? manifests.details[primaryIdx] ?? null
    : null

  const value = useMemo<WorkIdentity>(() => ({
    work,
    workId,
    primaryIdx,
    primaryDetail,
    manifestDetails:  manifests.details,
    manifestsLoading: manifests.loading,
    workCtx:          workCtxQ.data ?? null,
  }), [
    work, workId, primaryIdx, primaryDetail,
    manifests.details, manifests.loading, workCtxQ.data,
  ])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
