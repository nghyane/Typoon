// Per-chapter state hook.
//
// Reads the work's chapter-state Map via context, returns the snapshot
// for a single chapterRef. Falls back to IDLE_CHAPTER_STATE when the
// chapter has neither archive nor job.
//
// Components mounting many of these (one per row) never open IDB —
// the Map is already in memory via WorkChaptersContext's bulk query.

import { useWorkChapters } from '../contexts/WorkChaptersContext'
import { getChapterState } from '../data/selectors/chapterState'
import type { ChapterState } from '../data/types'


export function useChapterState(chapterRef: string): ChapterState {
  const { chapterStateMap } = useWorkChapters()
  return getChapterState(chapterStateMap, chapterRef)
}
