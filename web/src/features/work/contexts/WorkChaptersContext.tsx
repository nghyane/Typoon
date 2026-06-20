// WorkChaptersContext — chapter-list data layer.
//
// Composes local archive/history queries + the merged
// chapter spine into a single context. Chapter rows read state via
// O(1) Map lookups instead of mounting per-row hooks (which would
// cost N × 2 IDB cursors on a 4k-chapter work).
//
// Reactivity: any IDB write to archives/history triggers a single
// emit at this level. The Map identity changes, downstream selectors
// recompute, but only consumers of changed Map entries actually need
// to react — rows use `getChapterState(map, ref)` which is stable per
// ref unless the underlying record changed.

import { createContext, useContext, useMemo, type ReactNode } from 'react'

import { useEnabledSources } from '@features/browse/sources'
import { useWorkArchives } from '../data/queries/useWorkArchives'
import { useWorkHistoryMap } from '../data/queries/useWorkHistoryMap'
import { buildChapterStateMap } from '../data/selectors/chapterState'
import { mergeChapters } from '../data/selectors/mergeChapters'
import { pickReadTarget, type ReadTarget } from '../data/selectors/readTarget'
import type {
  ChapterState, HistoryItem, MergedChapter, SourceChapterDetail,
} from '../data/types'

import { useWorkIdentity } from './WorkIdentityContext'


export interface WorkChapters {
  merged:           MergedChapter[]
  sourceChapters:   SourceChapterDetail[]
  chapterStateMap:  ReadonlyMap<string, ChapterState>
  historyMap:       ReadonlyMap<string, HistoryItem>
  /** Chapters merged across attached sources. */
  totalChapters:    number
  readTarget:       ReadTarget | null
  loading:          boolean
}


const Ctx = createContext<WorkChapters | null>(null)


export function useWorkChapters(): WorkChapters {
  const v = useContext(Ctx)
  if (!v) throw new Error('useWorkChapters must be used inside <WorkChaptersProvider>')
  return v
}


interface Props { children: ReactNode }


export function WorkChaptersProvider({ children }: Props) {
  const { work, workId, manifestDetails, manifestsLoading } = useWorkIdentity()
  const installed  = useEnabledSources()
  const archives   = useWorkArchives(workId)
  const historyMap = useWorkHistoryMap(workId)

  // Shape source × refs for the merger
  const sourceChapters = useMemo<SourceChapterDetail[]>(() => {
    const out: SourceChapterDetail[] = []
    work.sources.forEach((origin, i) => {
      const detail = manifestDetails[i]
      if (!detail) return
      const src = installed.find(s => s.manifest.id === origin.source) ?? null
      if (!src) return
      out.push({ source: src, origin, refs: detail.chapters })
    })
    return out
  }, [work.sources, manifestDetails, installed])

  const merged = useMemo(
    () => mergeChapters(sourceChapters, work.target_lang.toLowerCase()),
    [sourceChapters, work.target_lang],
  )

  // Build state map ONCE — only refs that actually have a local archive land here
  const chapterStateMap = useMemo(
    () => buildChapterStateMap(archives),
    [archives],
  )

  const readTarget = useMemo(
    () => pickReadTarget(historyMap, merged),
    [historyMap, merged],
  )

  const value = useMemo<WorkChapters>(() => ({
    merged,
    sourceChapters,
    chapterStateMap,
    historyMap,
    totalChapters: merged.length,
    readTarget,
    loading: manifestsLoading,
  }), [
    merged, sourceChapters, chapterStateMap, historyMap,
    readTarget, manifestsLoading,
  ])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
