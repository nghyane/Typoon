// useReader — composed hook driving the unified reader.
//
// Inputs from the URL: `workId` + `numberNorm` (+ optional `lang`,
// `src`). The hook:
//
//   1. fetches the Work payload (shared cache with detail page),
//   2. picks the right HubVersion for the requested chapter based on
//      user's lang preference: translation done > raw target-lang
//      > any raw,
//   3. dispatches to the right source query (translation archive /
//      manifest pages),
//   4. resolves prev/next neighbour chapters with the SAME preference
//      so navigation feels stable,
//   5. records reading history,
//   6. drives Discord presence + portrait lock.
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
import { useChapterArchive } from './useChapterArchive'
import { useTranslation, useChapterPages } from './queries'
import type {
  ReaderSource, ReaderPage, ReaderNavTarget,
} from './types'


export interface UseReaderInput {
  workId:     number
  numberNorm: string
  /** Reading-lang override. Defaults to viewer's entry preference. */
  lang?:      string
  /** Active source material override (carried in URL search). */
  src?:       number
}


export interface UseReaderResult extends ReaderSource {
  /** What the shell should render at the top level. `not-found` =
   *  chapter doesn't exist in this work; `pending-render` = picked a
   *  translation that hasn't finished rendering; `no-source` = raw
   *  picked but its source plugin isn't installed locally. */
  status:        'loading' | 'ready' | 'not-found' | 'pending-render'
                | 'no-source' | 'error'
  /** Which HubVersion the reader ultimately picked. Exposed so the
   *  toolbar can show the right language chip / Đọc raw / Đọc bản
   *  dịch label. */
  picked:        HubVersion | null
  /** The whole HubChapter so the toolbar / chapter-list panel can
   *  reach sibling versions (other translations, raw on a different
   *  source). */
  chapter:       HubChapter | null
}


export function useReader(input: UseReaderInput): UseReaderResult {
  const { workId, numberNorm, lang: langOverride, src } = input

  const installed = useSources((s) => s.sources)
  const work = useWorkData(workId, src ?? null)
  const targetLang = (langOverride ?? work.targetLang ?? '').toLowerCase() || null

  // Locate the current HubChapter + pick the version the user
  // actually wants to read. Pure derivation — no extra queries.
  const { chapter, picked } = useMemo(() => {
    const ch = work.chapters.find((c) => c.number === numberNorm) ?? null
    return {
      chapter: ch,
      picked:  ch ? pickReadable(ch, targetLang) : null,
    }
  }, [work.chapters, numberNorm, targetLang])

  // Prev/next neighbours, resolved with the same preference. The
  // toolbar links to `/r/$workId/$numberNorm` so the URL is the
  // source of truth across chapter switches.
  const nav = useMemo(() => {
    if (!chapter) return { prev: null, next: null }
    return resolveNav(work.chapters, chapter, workId, targetLang, src)
  }, [work.chapters, chapter, workId, targetLang, src])

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
      return rawPagesQ.data.pages.map((u, i) => ({
        index:  i,
        url:    proxify(u),
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
  const workTitle = work.activeMaterial?.title
                 ?? trans?.material_title
                 ?? ''
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
      ? 'not-found'
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
    urls: isTranslation ? archive.urls : undefined,
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
    chapter,
  }
}


// ── Pure resolvers ────────────────────────────────────────────


/** Pick the best HubVersion for the requested lang. Priority:
 *    1. translation `done` in target lang
 *    2. raw whose lang matches target (read source verbatim)
 *    3. any spawnable raw
 *    4. anything (last resort).
 *
 *  Returns null only for empty chapters. */
function pickReadable(
  ch:         HubChapter,
  targetLang: string | null,
): HubVersion | null {
  const lang = targetLang?.toLowerCase().split(/[-_]/)[0] ?? null

  if (lang) {
    const tx = ch.versions.find(
      (v) => v.kind === 'translation'
          && v.lang === lang
          && v.state === 'done',
    )
    if (tx) return tx
  }

  if (lang) {
    const rawTgt = ch.versions.find(
      (v) => v.kind === 'raw' && v.lang === lang && !!v.upstreamUrl,
    )
    if (rawTgt) return rawTgt
  }

  const rawAny = ch.versions.find(
    (v) => v.kind === 'raw' && !!v.upstreamUrl,
  )
  if (rawAny) return rawAny

  return ch.versions[0] ?? null
}


/** Find the next / previous chapter that has at least one readable
 *  version under the same lang preference. The chapter spine is
 *  sorted latest-first (descending sortKey) — idx-1 is newer
 *  ("next" semantically) and idx+1 is older ("previous"). */
function resolveNav(
  chapters:   HubChapter[],
  current:    HubChapter,
  workId:     number,
  targetLang: string | null,
  src:        number | undefined,
): { prev: ReaderNavTarget | null; next: ReaderNavTarget | null } {
  const idx = chapters.findIndex((c) => c.number === current.number)
  if (idx < 0) return { prev: null, next: null }

  const find = (start: number, step: -1 | 1): ReaderNavTarget | null => {
    for (let i = start; i >= 0 && i < chapters.length; i += step) {
      const c = chapters[i]!
      if (pickReadable(c, targetLang)) {
        return {
          workId,
          numberNorm: c.number,
          lang:       targetLang ?? undefined,
          src,
        }
      }
    }
    return null
  }

  return {
    next: find(idx - 1, -1),
    prev: find(idx + 1, +1),
  }
}
