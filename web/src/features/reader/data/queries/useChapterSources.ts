// useChapterSources — bulk-read everything available for one chapter.
//
// Wraps:
//   saved (IDB live)
//   versions (from work's merged chapter spine, no extra fetch)
//
// Output is reactive; the picker UI reads this to render its list,
// the resolver reads this to decide an ActiveSource.

import { useLiveQuery } from 'dexie-react-hooks'
import { useMemo } from 'react'

import { db } from '@shared/db'
import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'
import type {
  ChapterSources, SavedArchive,
} from '../types'


export function useChapterSources(
  workId:     string,
  chapterRef: string,
): ChapterSources {
  const { merged } = useWorkChapters()

  // Saved blob — single IDB row.
  const saved = useLiveQuery<SavedArchive | null>(
    async () => (await db().archives.get(`${workId}:${chapterRef}`)) ?? null,
    [workId, chapterRef],
  ) ?? null

  // Versions live in the work's merged spine — no extra IDB read.
  const versions = useMemo(() => {
    const ch = merged.find(c => c.numberNorm === chapterRef)
    return ch?.sourceVersions ?? []
  }, [merged, chapterRef])

  return useMemo<ChapterSources>(() => ({
    saved, versions,
  }), [saved, versions])
}
