// Work resolution + lifecycle hooks.
//
// A Work is the stable identity object: created on first contact (browse
// or paste), promoted to library by an explicit user action, mutated by
// the source-attach flow. This module owns get-or-create logic so the
// rest of the app never sees a missing work_id.
//
// TanStack Query wraps Dexie reads to get React reactivity through the
// existing query-key invalidation pattern. Mutations invalidate the
// minimal set of keys they touch.

import { useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { nanoid } from 'nanoid'

import {
  db,
  deriveSourceKeys,
  sourceKey,
  type Work,
  type WorkSource,
  type LibraryStatus,
} from '@shared/db'
import { qk } from '@shared/api/keys'
import { pickPrimarySourceIndex } from '@features/work/data/selectors/primarySource'

export type { Work, WorkSource, LibraryStatus }


/** Refresh `work.title` + `cover_url` from the current primary source
 *  unless the user has explicitly overridden them. Returns a partial
 *  patch — callers spread it into the next Work. */
function syncIdentityToPrimary(
  cur:    Work,
  next:   WorkSource[],
): Partial<Pick<Work, 'title' | 'cover_url'>> {
  const idx = pickPrimarySourceIndex(next, cur.target_lang)
  if (idx < 0) return {}
  const primary = next[idx]!
  const patch: Partial<Pick<Work, 'title' | 'cover_url'>> = {}

  if (!cur.title_overridden) {
    const t = primary.title.trim()
    if (t && t !== cur.title) patch.title = t
  }

  if (!cur.cover_overridden) {
    if (primary.cover_url !== cur.cover_url) {
      patch.cover_url = primary.cover_url
    }
  }

  return patch
}


// ── Reads ───────────────────────────────────────────────────────────

export function useWork(workId: string | null | undefined) {
  return useQuery({
    queryKey: workId ? qk.works.byId(workId) : qk.works.invalid(),
    queryFn:  () => db().works.get(workId!).then(w => w ?? null),
    enabled:  !!workId,
    staleTime: Infinity,
  })
}

/** Find a Work that already has a matching `(source, upstream_ref)` in
 *  its `sources[]`. Backed by the multi-entry `*sourceKey` index, so
 *  this is O(log n) on a single Dexie key lookup. */
export function useWorkBySourceRef(
  source:       string | null | undefined,
  upstream_ref: string | null | undefined,
) {
  return useQuery({
    queryKey: qk.works.bySourceRef(source, upstream_ref),
    queryFn:  async () => {
      if (!source || !upstream_ref) return null
      const key = sourceKey(source, upstream_ref)
      const matches = await db().works.where('sourceKey').equals(key).toArray()
      return matches.find(w => !w.deleted) ?? null
    },
    enabled:  !!source && !!upstream_ref,
    staleTime: Infinity,
  })
}

/** Recently opened works (browse-only allowed). Used by Home /
 *  "Tiếp tục đọc" rail and by the LRU prune scheduler. */
export function useRecentlyOpened(limit = 30) {
  return useQuery({
    queryKey: qk.works.recent(),
    queryFn:  () =>
      db().works
        .orderBy('last_opened_at')
        .reverse()
        .filter(w => !w.deleted)
        .limit(limit)
        .toArray(),
    staleTime: Infinity,
  })
}

/** Resolve `work_id → Work` for a small set of ids. Used by Jobs lists
 *  that need to surface titles regardless of library state. */
export function useWorksByIds(ids: ReadonlyArray<string | null | undefined>) {
  const key = [...new Set(ids.filter((i): i is string => !!i))].sort()
  return useQuery({
    queryKey: ['works', 'by-ids', key],
    queryFn:  async () => {
      if (key.length === 0) return new Map<string, Work>()
      const rows = await db().works.bulkGet(key)
      const map = new Map<string, Work>()
      for (const w of rows) if (w && !w.deleted) map.set(w.id, w)
      return map
    },
    staleTime: 30_000,
  })
}


// ── Mutations ───────────────────────────────────────────────────────

export interface WorkSnapshot {
  title:        string
  cover_url?:   string | null
  source_lang?: string
  target_lang?: string
  nsfw?:        boolean
  /** First source's languages, when known. Falls back to `[source_lang]`. */
  languages?:   string[]
}

/** Get-or-create a Work for a `(source, upstream_ref)` pair. Creates
 *  with `in_library=false` so this flow stays browse-safe — promotion
 *  is a separate, explicit step. Touches `last_opened_at` on every call. */
export function useEnsureWorkFromSource() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: {
      source:       string
      upstream_ref: string
      snapshot:     WorkSnapshot
    }): Promise<Work> => {
      const now = new Date().toISOString()
      const key = sourceKey(args.source, args.upstream_ref)

      // Multi-entry index lookup — atomic enough for our single-tab use.
      const existing = (await db().works.where('sourceKey').equals(key).toArray())
        .find(w => !w.deleted)

      if (existing) {
        const next: Work = { ...existing, last_opened_at: now }
        await db().works.put(next)
        return next
      }

      const source_lang = args.snapshot.source_lang ?? 'ja'
      const target_lang = args.snapshot.target_lang ?? 'vi'
      const newSource: WorkSource = {
        source:       args.source,
        upstream_ref: args.upstream_ref,
        title:        args.snapshot.title,
        cover_url:    args.snapshot.cover_url ?? null,
        languages:    args.snapshot.languages ?? [source_lang],
        added_at:     now,
      }
      const sources = [newSource]
      const work: Work = {
        id:               nanoid(12),
        title:            args.snapshot.title,
        cover_url:        args.snapshot.cover_url ?? null,
        source_lang,
        target_lang,
        nsfw:             !!args.snapshot.nsfw,
        sources,
        sourceKey:        deriveSourceKeys(sources),
        in_library:       false,
        library_status:   null,
        library_added_at: null,
        last_opened_at:   now,
        created_at:       now,
        updated_at:       now,
      }
      await db().works.put(work)
      return work
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
      qc.invalidateQueries({ queryKey: qk.works.recent() })
      qc.invalidateQueries({ queryKey: qk.library.all() })
      // Source-ref lookups must reflect the new attachment.
      qc.invalidateQueries({ queryKey: ['works', 'by-source-ref'] })
    },
  })
}

/** Create a blank Work with no sources — used for upload-only flows. */
export function useCreateBlankWork() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (input: {
      title:        string
      source_lang?: string
      target_lang?: string
      nsfw?:        boolean
    }): Promise<Work> => {
      const now = new Date().toISOString()
      const work: Work = {
        id:               nanoid(12),
        title:            input.title,
        cover_url:        null,
        source_lang:      input.source_lang ?? 'ja',
        target_lang:      input.target_lang ?? 'vi',
        nsfw:             !!input.nsfw,
        sources:          [],
        sourceKey:        [],
        in_library:       false,
        library_status:   null,
        library_added_at: null,
        last_opened_at:   now,
        created_at:       now,
        updated_at:       now,
      }
      await db().works.put(work)
      return work
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
      qc.invalidateQueries({ queryKey: qk.works.recent() })
      qc.invalidateQueries({ queryKey: qk.library.all() })
    },
  })
}

/** Append a new source to an existing Work. No-op if the
 *  `(source, upstream_ref)` already attaches. Also re-evaluates the
 *  display title/cover when the user hasn't explicitly renamed — a
 *  newly-attached source may now be the better primary (e.g. user VN
 *  attaches Otruyen VI after MangaDex JP). */
export function useAttachSource() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: {
      work_id:      string
      source:       WorkSource
    }): Promise<Work> => {
      const cur = await db().works.get(args.work_id)
      if (!cur) throw new Error('Work không tồn tại.')
      const exists = cur.sources.some(s =>
        s.source === args.source.source && s.upstream_ref === args.source.upstream_ref,
      )
      if (exists) return cur

      const now = new Date().toISOString()
      const sources = [...cur.sources, args.source]
      const next: Work = {
        ...cur,
        ...syncIdentityToPrimary(cur, sources),
        sources,
        sourceKey:  deriveSourceKeys(sources),
        updated_at: now,
      }
      await db().works.put(next)
      return next
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
      qc.invalidateQueries({ queryKey: qk.library.all() })
      qc.invalidateQueries({ queryKey: ['works', 'by-source-ref'] })
    },
  })
}

/** Detach a source from a Work. If it was the last source, the Work
 *  stays as a blank (still browse-able, still in library if pinned).
 *  Also re-evaluates display title/cover when the user hasn't
 *  renamed — removing the current primary should not leave its
 *  stale title cached. */
export function useDetachSource() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: {
      work_id:      string
      source:       string
      upstream_ref: string
    }): Promise<Work> => {
      const cur = await db().works.get(args.work_id)
      if (!cur) throw new Error('Work không tồn tại.')
      const sources = cur.sources.filter(s =>
        !(s.source === args.source && s.upstream_ref === args.upstream_ref),
      )
      const next: Work = {
        ...cur,
        ...syncIdentityToPrimary(cur, sources),
        sources,
        sourceKey:  deriveSourceKeys(sources),
        updated_at: new Date().toISOString(),
      }
      await db().works.put(next)
      return next
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
      qc.invalidateQueries({ queryKey: qk.library.all() })
      qc.invalidateQueries({ queryKey: ['works', 'by-source-ref'] })
    },
  })
}

export function useUpdateWork() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (args: {
      id:    string
      patch: Partial<Pick<Work,
        'title' | 'title_overridden' | 'cover_url' | 'cover_overridden'
        | 'nsfw' | 'source_lang' | 'target_lang'
      >>
    }): Promise<Work> => {
      const cur = await db().works.get(args.id)
      if (!cur) throw new Error('Work không tồn tại.')

      // Merge user patch first so the override flags reflect the
      // user's just-made choice before we run the auto-sync.
      const merged: Work = { ...cur, ...args.patch }

      // If the caller dropped an override flag, re-pull from the
      // primary source so the user sees the canonical value
      // immediately instead of waiting for the next attach/detach.
      const droppedTitle = args.patch.title_overridden === false
      const droppedCover = args.patch.cover_overridden === false
      const sync = (droppedTitle || droppedCover)
        ? syncIdentityToPrimary(merged, merged.sources)
        : {}

      const next: Work = {
        ...merged,
        ...sync,
        updated_at: new Date().toISOString(),
      }
      await db().works.put(next)
      return next
    },
    onSuccess: (work) => {
      qc.invalidateQueries({ queryKey: qk.works.byId(work.id) })
      qc.invalidateQueries({ queryKey: qk.library.all() })
    },
  })
}

/** Update `last_opened_at` without other mutations. Lightweight — used
 *  on route enter to drive Recent-opened ordering. */
export function useTouchWork() {
  const qc = useQueryClient()
  return useCallback(async (workId: string) => {
    const cur = await db().works.get(workId)
    if (!cur) return
    await db().works.update(workId, { last_opened_at: new Date().toISOString() })
    qc.invalidateQueries({ queryKey: qk.works.recent() })
  }, [qc])
}
