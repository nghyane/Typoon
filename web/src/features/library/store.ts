// Library — cross-source bookmark + reading history (local-first).
//
// One entry per (source, mangaUrl). Same row carries:
//   • bookmark flag (Hội Mê Truyện / browse hub / library list)
//   • last-read tracking (for "Tiếp tục đọc" rails and library list)
//   • latest-chapter snapshot (so we can derive `hasNew` without server)
//
// Persisted to localStorage. Refresh happens browser-side via the DA
// proxy (free egress) — no backend tables, no worker. See
// docs/wiki/browse-mode.md §5 for the history doctrine this extends.
//
// Migration: on first run we read the legacy `typoon.readingHistory.v1`
// blob, map it onto the new shape, then delete the old key. After
// migration the legacy module is dead and gets removed in the same PR.

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

const LIBRARY_MAX = 200

const LEGACY_HISTORY_KEY = 'typoon.readingHistory.v1'

export interface ChapterRef {
  url:    string
  label:  string
  number: string
}

export interface LibraryEntry {
  /** Identity */
  source:    string         // manifest.id
  mangaUrl:  string         // upstream URL (route param)
  title:     string
  cover:     string | null

  /** Bookmark side */
  bookmarked:    boolean
  bookmarkedAt:  number | null

  /** Reading side */
  lastReadAt:        number | null
  lastChapterRead:   ChapterRef | null

  /** Latest chapter seen at source (snapshot from last manga detail
   *  fetch). `hasNew` derives from `latestChapter.url !== lastChapterRead.url`. */
  latestChapter:     ChapterRef | null
  latestSeenAt:      number | null
}

export function hasNewChapter(e: LibraryEntry): boolean {
  return !!(e.latestChapter && e.lastChapterRead
    && e.latestChapter.url !== e.lastChapterRead.url)
}

interface LibraryStore {
  items: Record<string, LibraryEntry>  // key = `${source}::${mangaUrl}`

  /** Called on every manga-detail fetch. Upserts identity + cover +
   *  latestChapter (chapters[0]). Does NOT touch `lastReadAt` or
   *  bookmark flag. */
  recordView: (args: {
    source:   string
    mangaUrl: string
    title:    string
    cover:    string | null
    latestChapter: ChapterRef | null
  }) => void

  /** Called when user opens a chapter in the reader. Updates
   *  lastReadAt + lastChapterRead. Creates entry if missing
   *  (defensive — usually recordView ran first). */
  markChapterRead: (args: {
    source:   string
    mangaUrl: string
    title:    string
    cover:    string | null
    chapter:  ChapterRef
  }) => void

  /** Toggle bookmark. Creates entry if missing. */
  toggleBookmark: (args: {
    source:   string
    mangaUrl: string
    title:    string
    cover:    string | null
  }) => void

  /** Remove entry entirely. */
  remove: (source: string, mangaUrl: string) => void

  /** Drop entries beyond LIBRARY_MAX, oldest unbookmarked first.
   *  Bookmarked entries are immune to capacity eviction. */
  prune: () => void

  clear: () => void
}

function key(source: string, mangaUrl: string): string {
  return `${source}::${mangaUrl}`
}

function defaultEntry(args: {
  source:   string
  mangaUrl: string
  title:    string
  cover:    string | null
}): LibraryEntry {
  return {
    source:           args.source,
    mangaUrl:         args.mangaUrl,
    title:            args.title,
    cover:            args.cover,
    bookmarked:       false,
    bookmarkedAt:     null,
    lastReadAt:       null,
    lastChapterRead:  null,
    latestChapter:    null,
    latestSeenAt:     null,
  }
}

/** Migrate legacy reading history (one-shot). Runs only if the legacy
 *  blob is present and the new store hasn't persisted anything yet. */
function migrateLegacy(): Record<string, LibraryEntry> {
  if (typeof window === 'undefined') return {}
  const raw = localStorage.getItem(LEGACY_HISTORY_KEY)
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw) as {
      state?: { items?: Array<{
        source:   string
        mangaUrl: string
        title:    string
        cover:    string | null
        chapter?: { url: string; label: string; number: string }
        updatedAt: number
      }> }
    }
    const out: Record<string, LibraryEntry> = {}
    for (const it of parsed.state?.items ?? []) {
      out[key(it.source, it.mangaUrl)] = {
        source:     it.source,
        mangaUrl:   it.mangaUrl,
        title:      it.title,
        cover:      it.cover,
        bookmarked: false,
        bookmarkedAt: null,
        lastReadAt: it.updatedAt,
        lastChapterRead: it.chapter ?? null,
        latestChapter: null,
        latestSeenAt: null,
      }
    }
    return out
  } catch {
    return {}
  } finally {
    // Always drop the legacy key — even on parse failure — so the
    // migration doesn't keep retrying on every load.
    localStorage.removeItem(LEGACY_HISTORY_KEY)
  }
}

export const useLibrary = create<LibraryStore>()(
  persist(
    (set, get) => ({
      items: {},

      recordView: ({ source, mangaUrl, title, cover, latestChapter }) => {
        const k = key(source, mangaUrl)
        const cur = get().items[k] ?? defaultEntry({ source, mangaUrl, title, cover })
        const next: LibraryEntry = {
          ...cur,
          // Refresh identity fields — title/cover can change upstream.
          title,
          cover,
          latestChapter: latestChapter ?? cur.latestChapter,
          latestSeenAt: latestChapter ? Date.now() : cur.latestSeenAt,
        }
        // Skip write if nothing changed (avoid storage churn on
        // every MangaPage focus).
        if (
          cur.title === next.title
          && cur.cover === next.cover
          && cur.latestChapter?.url === next.latestChapter?.url
        ) return
        set({ items: { ...get().items, [k]: next } })
      },

      markChapterRead: ({ source, mangaUrl, title, cover, chapter }) => {
        const k = key(source, mangaUrl)
        const cur = get().items[k] ?? defaultEntry({ source, mangaUrl, title, cover })
        const now = Date.now()
        const next: LibraryEntry = {
          ...cur,
          title,
          cover,
          lastReadAt:      now,
          lastChapterRead: chapter,
          // Opening a chapter implicitly "acknowledges" any new-chapter
          // signal — if the just-read chapter happens to be the latest,
          // hasNew flips off naturally. We do NOT clear latestChapter
          // here; let recordView do that on the next manga-page mount.
        }
        const items = { ...get().items, [k]: next }
        // Cheap LRU prune when entry count crosses the cap.
        const over = Object.keys(items).length - LIBRARY_MAX
        if (over > 0) {
          const candidates = Object.values(items)
            .filter((e) => !e.bookmarked)
            .sort((a, b) => (a.lastReadAt ?? 0) - (b.lastReadAt ?? 0))
            .slice(0, over)
          for (const e of candidates) delete items[key(e.source, e.mangaUrl)]
        }
        set({ items })
      },

      toggleBookmark: ({ source, mangaUrl, title, cover }) => {
        const k = key(source, mangaUrl)
        const cur = get().items[k] ?? defaultEntry({ source, mangaUrl, title, cover })
        const turnOn = !cur.bookmarked
        const next: LibraryEntry = {
          ...cur,
          title,
          cover,
          bookmarked:   turnOn,
          bookmarkedAt: turnOn ? Date.now() : null,
        }
        set({ items: { ...get().items, [k]: next } })
      },

      remove: (source, mangaUrl) => {
        const k = key(source, mangaUrl)
        const next = { ...get().items }
        delete next[k]
        set({ items: next })
      },

      prune: () => {
        const items = get().items
        const arr = Object.values(items)
        if (arr.length <= LIBRARY_MAX) return
        const over = arr.length - LIBRARY_MAX
        const candidates = arr
          .filter((e) => !e.bookmarked)
          .sort((a, b) => (a.lastReadAt ?? 0) - (b.lastReadAt ?? 0))
          .slice(0, over)
        const out = { ...items }
        for (const e of candidates) delete out[key(e.source, e.mangaUrl)]
        set({ items: out })
      },

      clear: () => set({ items: {} }),
    }),
    {
      name:    'typoon.library.v1',
      storage: createJSONStorage(() => localStorage),
      // Merge migrated legacy entries on first hydrate. The migration
      // function deletes the old key, so this only does work once.
      merge: (persisted, current) => {
        const persistedState = (persisted as { items?: Record<string, LibraryEntry> }) ?? {}
        const persistedItems = persistedState.items ?? {}
        const legacyItems = migrateLegacy()
        return {
          ...current,
          // Persisted wins over legacy if the same key exists in both
          // (user has interacted with new store since migration started).
          items: { ...legacyItems, ...persistedItems },
        }
      },
    },
  ),
)
