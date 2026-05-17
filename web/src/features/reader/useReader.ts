// useReader — composed hook driving the unified reader.
//
// Inputs from the URL: `workId` + `numberNorm`.
// The hook:
//
//   1. fetches the Work payload (shared cache with detail page),
//   2. picks the right HubVersion for the requested chapter:
//      saved source pref → default fallback (translation done in
//      target lang → raw target → any raw),
//   3. dispatches to the right source query (translation archive /
//      manifest pages),
//   4. resolves prev/next neighbour chapters with the SAME preference
//      so navigation auto-follows the user's chosen source,
//   5. records reading history,
//   6. drives Discord presence + portrait lock.
//
// The Work payload auto-merges chapters across every installed
// source, so the reader never needs a per-source URL param — the
// per-work `SourcePreference` (set the moment the user taps Đọc
// in the picker) is the only thing that decides which version
// renders, and it sticks across chapter switches.
//
// Returns a `ReaderSource` shape the `<Reader>` shell consumes
// without caring which kind of pixels backed it.

import { useEffect, useMemo, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api } from '@shared/api/api'
import {
  clearReadingPresence, lockPortrait, setReadingPresence, unlockOrientation,
} from '@shared/discord/presence'
import { proxify } from '@features/browse/proxy'
import { useSources } from '@features/browse/sources'
import type { HubChapter, HubVersion } from '@features/title/mergeChapters'

import { useWorkData } from '@features/work/useWorkData'
import { pickPrimaryMaterial } from '@features/work/title'
import { useChapterArchive } from './useChapterArchive'
import { useTranslation, useChapterPages } from './queries'
import { resolvePicked, pickByPref, resolveNav } from './resolvers'
import { useReaderSettings, sourcePrefFor } from './store'
import type {
  ReaderSource, ReaderPage,
} from './types'


export interface UseReaderInput {
  workId:     number
  numberNorm: string
}


export interface UseReaderResult extends ReaderSource {
  /** What the shell should render at the top level. `not-found` =
   *  chapter doesn't exist in this work; `pending-render` = picked a
   *  translation that hasn't finished rendering; `no-source` = raw
   *  picked but its source plugin isn't installed locally; `empty` =
   *  the chapter exists in the spine but has no readable nor
   *  spawnable version (a filler / cover the user can choose to
   *  skip — different from `not-found` which means the slug is bogus). */
  status:        'loading' | 'ready' | 'not-found' | 'pending-render'
                | 'no-source' | 'empty' | 'error'
  /** Which HubVersion the reader ultimately picked. Exposed so the
   *  toolbar can show the right language chip / Đọc raw / Đọc bản
   *  dịch label. */
  picked:        HubVersion | null
  /** True when the user has a source pref set for this work AND the
   *  current chapter has no version matching that pref. Drives the
   *  in-reader banner "Chương này không có bản dịch từ EN…". */
  prefMismatch:  boolean
  /** The whole HubChapter so the toolbar / chapter-list panel can
   *  reach sibling versions (other translations, raw on a different
   *  source). */
  chapter:       HubChapter | null
}


export function useReader(input: UseReaderInput): UseReaderResult {
  const { workId, numberNorm } = input

  const installed = useSources((s) => s.sources)
  const work = useWorkData(workId)
  const targetLang = work.targetLang

  // Source preference — sticky per-work choice the user made via
  // the in-reader source picker (e.g. "AI VI từ EN"). Every Đọc
  // tap writes here, so navigating to the next chapter
  // automatically re-uses the same (kind, lang, sourceLang) tuple.
  const pref = useReaderSettings((s) => sourcePrefFor(s, workId))

  // Locate the current HubChapter + resolve the picked version via
  // pref → default fallback. When the chapter has no version
  // matching the saved pref we silently fall back to `pickReadable`
  // and surface `prefMismatch` so the route can render a banner.
  const { chapter, picked, prefMismatch } = useMemo(() => {
    const ch = work.chapters.find((c) => c.number === numberNorm) ?? null
    if (!ch) return { chapter: null, picked: null, prefMismatch: false }
    const p = resolvePicked(ch, targetLang, pref)
    const miss = pref !== null && pickByPref(ch, pref) === null
    return { chapter: ch, picked: p, prefMismatch: miss }
  }, [work.chapters, numberNorm, targetLang, pref])

  // Prev/next neighbours, resolved with the same preference. The
  // toolbar links to `/r/$workId/$numberNorm` so the URL is the
  // source of truth across chapter switches.
  const nav = useMemo(() => {
    if (!chapter) return { prev: null, next: null }
    return resolveNav(work.chapters, chapter, workId)
  }, [work.chapters, chapter, workId])

  // Dispatch to the right source query based on the picked version's
  // kind. Both queries `enable: false` until a version of the right
  // kind is in hand so we don't fire stray network calls.
  const isTranslation = picked?.kind === 'translation'
  const isRaw         = picked?.kind === 'raw'

  const translationQ = useTranslation(
    isTranslation ? picked!.translationId : null,
  )
  const trans = translationQ.data ?? null

  const archive = useChapterArchive(
    isTranslation ? trans?.archive_url : null,
  )

  const rawSource = isRaw && picked!.sourceId
    ? installed[picked!.sourceId] ?? null
    : null
  const rawPagesQ = useChapterPages(
    rawSource,
    isRaw ? picked!.upstreamUrl : null,
  )

  // Build the unified page array.
  const pages = useMemo<ReaderPage[]>(() => {
    if (isTranslation && archive.bunle) {
      return archive.bunle.pages.map((info) => ({
        index:  info.index,
        url:    null,  // streamed — read from `urls` map instead
        width:  info.width,
        height: info.height,
      }))
    }
    if (isRaw && rawPagesQ.data) {
      const tokens = rawPagesQ.data.tokens
      return rawPagesQ.data.pages.map((u, i) => ({
        index:  i,
        url:    tokens ? null : proxify(u),
        token:  tokens?.[i],
        width:  0,
        height: 0,
      }))
    }
    return []
  }, [isTranslation, archive.bunle, isRaw, rawPagesQ.data])

  // Reading history — fire-and-forget once per chapter resolve.
  const recordedRef = useRef<string | null>(null)
  const recordTrans = useMutation({ mutationFn: api.recordTranslatedReading })
  const recordRaw   = useMutation({ mutationFn: api.recordRawReading })

  useEffect(() => {
    if (!chapter || !picked) return
    if (isTranslation && trans) {
      const key = `t:${trans.id}`
      if (recordedRef.current === key) return
      recordedRef.current = key
      recordTrans.mutate({ translation_id: trans.id })
      return
    }
    if (isRaw && picked.materialId != null && picked.numberNorm) {
      const key = `r:${picked.materialId}:${picked.numberNorm}`
      if (recordedRef.current === key) return
      recordedRef.current = key
      recordRaw.mutate({
        material_id: picked.materialId,
        number:      chapter.number,
        number_norm: picked.numberNorm,
        label:       chapter.label,
      })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps -- mutations are stable
  }, [chapter, picked, isTranslation, isRaw, trans?.id])

  // Discord presence + portrait lock — same lifecycle for every kind.
  // The Work payload no longer carries an "active material"; pick a
  // primary one for the title here (same resolver as the hero would
  // do) and fall back to the translation's own snapshot when the
  // payload hasn't landed yet.
  const workTitle = useMemo(() => {
    const primary = pickPrimaryMaterial(work.materials, targetLang)
    return primary?.title ?? trans?.material_title ?? ''
  }, [work.materials, targetLang, trans?.material_title])
  useEffect(() => {
    if (!workTitle || !chapter) return
    setReadingPresence({
      projectTitle:  workTitle,
      chapterNumber: chapter.number,
      chapterTitle:  chapter.label,
    })
  }, [workTitle, chapter])
  useEffect(() => {
    lockPortrait()
    return () => {
      unlockOrientation()
      clearReadingPresence()
    }
  }, [])

  // Status — the shell renders different UI per state.
  //
  // For translated chapters the archive stream is a second async step
  // AFTER the translation row resolves: the row gives us `archive_url`,
  // then `useChapterArchive` fetches the index + streams blobs. We
  // must surface 'loading' until the bunle is open and at least one
  // page is ready, otherwise the user sees an empty page area between
  // chapters.
  const archiveReady = !!archive.bunle && archive.urls.size > 0
  const status: UseReaderResult['status'] =
    work.workLoading
      ? 'loading'
    : !chapter
      ? 'not-found'
    : !picked
      // Chapter exists in the spine but every version is unreadable
      // (raws without an installed source plugin or upstream URL,
      // translations not yet done). Surface as `empty` so the reader
      // can render a "filler / skip to next" affordance instead of
      // the 404 page that `not-found` triggers.
      ? 'empty'
    : isTranslation
      ? (translationQ.isPending
          ? 'loading'
        : !trans?.archive_url
          ? 'pending-render'
        : archive.error
          ? 'error'
        : !archiveReady
          ? 'loading'
        : 'ready')
    : isRaw
      ? (!rawSource
          ? 'no-source'
        : rawPagesQ.isPending
          ? 'loading'
        : rawPagesQ.isError
          ? 'error'
        : 'ready')
    : 'error'

  return {
    pages,
    urls:      isTranslation ? archive.urls : undefined,
    rawSource: isRaw ? rawSource ?? undefined : undefined,
    meta: {
      workId,
      workTitle,
      chapterText: chapter ? `Ch.${chapter.number}` : '',
      chapterSub:  chapter?.label ?? null,
      lang:        picked?.lang ?? targetLang,
    },
    nav,
    loading: status === 'loading',
    error:   archive.error ?? (rawPagesQ.error as Error | null)?.message ?? null,
    status,
    picked,
    prefMismatch,
    chapter,
  }
}
