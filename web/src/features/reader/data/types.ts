// Reader source data types.
//
// `SourcePref` is the user's sticky preference per work — what they
// picked once should apply to subsequent chapters.
//
// `ChapterSources` enumerates everything available for one chapter:
//   - saved offline blob (if any)
//   - raw source versions (from manifest)
//
// `ActiveSource` is the resolved decision after applying pref to
// the chapter's available sources. It's the unit the cache pool
// keys on, the unit the page renderer consumes.

import type { SavedArchive } from '@shared/db'
import type { SourceVersion } from '@features/work/data/types'
import type { ReaderSource, ReaderMode } from '../sources'

export type { ReaderSource, ReaderMode, SavedArchive, SourceVersion }


// ── User preference (sticky per work) ──────────────────────────


export type SourcePref =
  | { kind: 'auto' }                              // resolver picks best
  | { kind: 'raw'; versionKey: string }           // explicit raw source


export const DEFAULT_PREF: SourcePref = { kind: 'auto' }


// ── Available sources for one chapter ──────────────────────────


export interface ChapterSources {
  /** Saved offline blob (any kind). */
  saved:        SavedArchive | null
  /** Raw versions from manifest. */
  versions:     SourceVersion[]
}


// ── Resolved active source ─────────────────────────────────────


export type ActiveSource =
  | { kind: 'none' }
  | {
      kind:       'raw-offline'
      archiveId:  string
      blob:       Blob
    }
  | {
      kind:        'raw-online'
      versionKey:  string
      urls:        string[]
    }


// ── Reader source state machine ────────────────────────────────


export type ReaderSourceState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; source: ReaderSource; mode: ReaderMode }
  | { status: 'error'; error: Error }
  | { status: 'no-source' }
