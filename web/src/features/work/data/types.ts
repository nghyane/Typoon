// Work page data types — shared between data layer, selectors,
// contexts, and components.

import type {
  Work, WorkSource, SavedArchive, HistoryItem,
} from '@shared/db'
import type {
  InstalledSource, MangaChapterRef, MangaDetail,
} from '@features/browse/manifest/types'

export type {
  Work, WorkSource, SavedArchive, HistoryItem,
  InstalledSource, MangaChapterRef, MangaDetail,
}


// ── Chapter ────────────────────────────────────────────────────


/** One source's offering for a chapter. Same chapter may have N
 *  versions across sources (mdex EN, otruyen VI). */
export interface SourceVersion {
  source: InstalledSource
  origin: WorkSource
  ref:    MangaChapterRef
  /** Normalized lower-case lang code. */
  lang:   string
}


/** A chapter after cross-source merge by `numberNorm`. */
export interface MergedChapter {
  numberNorm: string
  label:      string
  number:     string
  sortKey:    number
  sourceVersions: SourceVersion[]
}


/** Source × refs bundle the chapter merger consumes. */
export interface SourceChapterDetail {
  source: InstalledSource
  origin: WorkSource
  refs:   MangaChapterRef[]
}


// ── Chapter state ──────────────────────────────────────────────


/** Computed status for one chapter. Order matters — first matching
 *  predicate wins. */
export type ChapterStatus =
  | 'idle'              // no saved archive
  | 'saved-raw'         // raw blob in IDB


/** Snapshot of one chapter's local persistence state. Pure data — no React,
 *  no fetchers. Derived once from the saved archive per chapter. */
export interface ChapterState {
  status:   ChapterStatus
  archive:  SavedArchive | null
}


export const IDLE_CHAPTER_STATE: ChapterState = {
  status:  'idle',
  archive: null,
}
