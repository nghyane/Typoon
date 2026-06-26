// Chapter state derivation — pure functions.
//
// Why pure: row components consume ChapterState as props. Keeping the
// derivation out of React means tests can verify it without rendering,
// and the same logic powers the reader resolver if needed later.
//
// Build strategy: only chapters that have an archive
// land in the map. "idle" chapters live nowhere — consumers fall back
// to `IDLE_CHAPTER_STATE` when `.get()` returns undefined. Keeps the
// map ≪ total chapters for typical works.

import type {
  ChapterState, ChapterStatus, SavedArchive,
} from '../types'
import { IDLE_CHAPTER_STATE } from '../types'


export function deriveChapterStatus(
  archive: SavedArchive | null,
): ChapterStatus {
  if (archive?.kind === 'raw')        return 'saved-raw'
  return 'idle'
}


export function deriveChapterState(
  archive: SavedArchive | null,
): ChapterState {
  return {
    status: deriveChapterStatus(archive),
    archive,
  }
}


/** Build a Map keyed by chapter_ref from the work's bulk archive
 *  collection. Only chapters with non-idle state are included. */
export function buildChapterStateMap(
  archives: ReadonlyMap<string, SavedArchive>,
): Map<string, ChapterState> {
  const map = new Map<string, ChapterState>()
  for (const [ref, archive] of archives) {
    map.set(ref, deriveChapterState(archive))
  }
  return map
}


/** Lookup helper: returns IDLE_CHAPTER_STATE when ref isn't in the map. */
export function getChapterState(
  map: ReadonlyMap<string, ChapterState>,
  ref: string,
): ChapterState {
  return map.get(ref) ?? IDLE_CHAPTER_STATE
}
