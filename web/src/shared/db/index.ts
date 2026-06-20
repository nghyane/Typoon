// Dexie schema v3 — work-centric, with library as a flag.
//
// Stores:
//   works    — every Work the user has touched (browse-only or pinned).
//              Pinning is a boolean flag (`in_library`), not a separate store,
//              so a single id survives "browse → read 1 chap → add to library"
//              with zero data movement.
//   history  — last-read position per (work, chapter). Independent of pinning.
//   archives — offline-saved chapter blobs.
//   settings — single-row user preferences blob.
//
// Identity model:
//   `Work.id` is a stable nanoid. It survives source add/remove because
//   server-side context is keyed on `(user, work_id)` — moving away
//   from "${source}:${upstream_ref}" composite ids was the whole point
//   of v3.5.
//
// The DB name `typoon-v3` is intentional: v3.5 schema is structurally
// incompatible with v1/v2, and we have no users with real data yet, so
// we orphan the old DB rather than write an upgrade fn.
//
// All timestamps are ISO strings (sortable, debuggable, identical across
// clients).

import Dexie, { type EntityTable } from 'dexie'


// ── Work types ──────────────────────────────────────────────────────

/** One source's manifestation of a Work — the same manga can be backed
 *  by N sources (Mangadex EN, Otruyen VI, …). Translation context is
 *  shared across all of them via the parent Work.id. */
export interface WorkSource {
  /** Adapter id (mdex, otruyen, hentaifox, …). */
  source:       string
  /** Upstream reference within the source — URL or stable id. */
  upstream_ref: string
  title:        string
  cover_url:    string | null
  languages:    string[]
  added_at:     string
}

export type LibraryStatus = 'reading' | 'plan' | 'done' | 'dropped'

export interface Work {
  /** Stable client-generated nanoid. Also keys per-work context on the backend. */
  id:            string
  /** Display title. Auto-synced to the primary source's title on
   *  attach/detach unless `title_overridden` is true (user renamed
   *  explicitly). Cached so list views (Library, Recent) render
   *  without loading every source manifest. */
  title:         string
  /** Sticky flag: set when the user renames the work by hand. Stops
   *  attach/detach from clobbering their choice. Optional/undefined =
   *  legacy false. */
  title_overridden?: boolean
  cover_url:     string | null
  /** Sticky flag: set when the user chose a custom cover (URL paste
   *  or custom image). Stops attach/detach from clobbering their choice.
   *  Same semantics as `title_overridden`. */
  cover_overridden?: boolean
  source_lang:   string
  target_lang:   string
  nsfw:          boolean

  /** Source materials backing this work. Empty array = manual library entry. */
  sources:       WorkSource[]

  /** Derived array of `"${source}:${upstream_ref}"`, mirrored from
   *  `sources[]` on every mutation. Used by the multi-entry index for
   *  O(1) lookup "given (source, ref), find work_id". Never read by
   *  application code — go through `useWorkBySourceRef` instead. */
  sourceKey:     string[]

  /** User pinned this work into their library? Browse-only works have
   *  `in_library=false` and are subject to LRU prune. */
  in_library:    boolean
  library_status:    LibraryStatus | null
  library_added_at:  string | null

  /** Touched on any visit (work hub, reader, source-resolve). Drives
   *  "Recently opened" + LRU prune. */
  last_opened_at: string

  created_at:    string
  updated_at:    string
  /** Tombstone for future sync. */
  deleted?:      boolean
}


// ── History ─────────────────────────────────────────────────────────

export interface HistoryItem {
  /** Composite key `${work_id}:${chapter_ref}`. */
  id:            string
  work_id:       string
  chapter_ref:   string
  /** Denormalized chapter label so the UI can show "Đang đọc · Ch. 12"
   *  without round-tripping the source manifest. */
  chapter_label: string
  page:          number
  total_pages:   number | null
  last_read_at:  string
}


// ── Settings ────────────────────────────────────────────────────────

export interface SettingsBlob {
  key:        'global'
  theme:      'system' | 'light' | 'dark'
  /** High-level reading style — 'standard' (LTR pager), 'rtl' (manga
   *  RTL pager), 'vertical' (LTR scroll inside pager), 'webtoon'
   *  (continuous strip). Legacy values 'pager' / 'strip' migrate to
   *  'rtl' / 'webtoon' on read. */
  reader_mode: 'standard' | 'rtl' | 'vertical' | 'webtoon' | 'pager' | 'strip'
  /** Reading direction (page order). Manga = rtl, manhua = ltr,
   *  webtoon = ttb. Global because most users settle on one. */
  reader_direction?: 'ltr' | 'rtl' | 'ttb'
  /** Image fit mode. width / height / free. */
  reader_fit?: 'width' | 'height' | 'free'
  /** Max page width in px for "free" / "width" fits. Default 1040,
   *  clamped 600..1600 at runtime. */
  reader_page_width?: number
  /** Pixel gap between pages in strip mode. 0..32. */
  reader_page_gap?: number
  /** Click left/right turn page (pager mode). */
  reader_click_turn?: boolean
  /** Preload N pages ahead of the current. 0..8. */
  reader_preload_ahead?: number
  /** Sticky source preference per work. Keyed by work_id.
   *  Survives chapter navigation so a user who picks "EN MangaDex"
   *  on chapter 42 stays on EN MangaDex for chapter 43. */
  reader_source_prefs?: Record<string, ReaderSourcePref>
  default_target_lang: string | null
  updated_at: string
}


export type ReaderSourcePref =
  | { kind: 'auto' }
  | { kind: 'raw'; versionKey: string }


// ── Archives ────────────────────────────────────────────────────────

export interface SavedArchive {
  /** Composite key `${work_id}:${chapter_ref}`. */
  id:          string
  work_id:     string
  chapter_ref: string
  kind:        'raw'
  blob:        Blob
  page_count:  number
  byte_size:   number
  saved_at:    string
}


// ── DB class ────────────────────────────────────────────────────────

export class TypoonDb extends Dexie {
  works!:    EntityTable<Work,         'id'>
  history!:  EntityTable<HistoryItem,  'id'>
  settings!: EntityTable<SettingsBlob, 'key'>
  archives!: EntityTable<SavedArchive, 'id'>

  constructor() {
    // v3.5 schema lives in a fresh DB name. Old `typoon` v1/v2 DBs are
    // orphaned; the browser will reclaim that storage on its own. We
    // explicitly do not write an upgrade fn because there are no users
    // with data worth preserving.
    super('typoon-v3')

    this.version(1).stores({
      works:    '&id, last_opened_at, updated_at, *sourceKey',
      history:  '&id, work_id, last_read_at',
      settings: '&key',
      archives: '&id, work_id, kind, saved_at',
    })

    // v2 — compound `[work_id+chapter_ref]` on archives.
    // Archive queries by both keys at once; without this, Dexie falls
    // back to a full scan + JS filter (works but logs a warning and is O(N)).
    this.version(2).stores({
      works:    '&id, last_opened_at, updated_at, *sourceKey',
      history:  '&id, work_id, last_read_at',
      settings: '&key',
      archives: '&id, work_id, [work_id+chapter_ref], kind, saved_at',
    })
  }
}


/** Lazily-instantiated singleton so SSR / test contexts don't open IDB
 *  at import time. Call from React/effects only. */
let _db: TypoonDb | null = null
export function db(): TypoonDb {
  if (!_db) _db = new TypoonDb()
  return _db
}


// ── Derived helpers ─────────────────────────────────────────────────

export function sourceKey(source: string, upstream_ref: string): string {
  return `${source}:${upstream_ref}`
}

export function deriveSourceKeys(sources: WorkSource[]): string[] {
  return sources.map(s => sourceKey(s.source, s.upstream_ref))
}
