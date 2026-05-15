// Reader route — /r/$workId/$numberNorm
//
// Thin route layer: parse URL params, hand off to `useReader`,
// render the shell. The shell is composed of:
//
//   ReaderTopBar         slim title strip (top, fixed)
//   ReaderBody           Strip OR Pager + TapZones + HoverEdges
//   ReaderBottomPill     floating control pill (centered)
//   SettingsSheet        tabbed modal
//   ChapterPicker         dropdown anchored to top-bar chapter trigger
//
// State is layered:
//   - URL:           workId, numberNorm, ?page (pager only)
//   - useReader:     chapter + pages from server
//   - useReaderUiState: chrome / sheets / peek
//   - useReaderSettings: persistent prefs (Zustand persist)
//   - localStorage  via useReaderPosition: resume per-chapter

import { createFileRoute, redirect as routerRedirect } from '@tanstack/react-router'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle } from 'lucide-react'

import { api, WorkRedirectedError } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { toast } from '@shared/ui/Toaster'

import { ReaderTopBar } from '@features/reader/ReaderTopBar'
import { ReaderBottomPill } from '@features/reader/ReaderBottomPill'
import { ReaderBody } from '@features/reader/Reader'
import { SettingsSheet } from '@features/reader/SettingsSheet'
import { ChapterPicker } from '@features/reader/ChapterPicker'
import { SourcePicker } from '@features/reader/SourcePicker'
import { EndOfChapterCard } from '@features/reader/EndOfChapterCard'
import { useReader } from '@features/reader/useReader'
import { useReaderUiState } from '@features/reader/useReaderUiState'
import { normalizeLang as normalizeBcp } from '@features/reader/resolvers'
import { useReaderKeyboard } from '@features/reader/useReaderKeyboard'
import {
  loadPosition, useReadingPosition,
} from '@features/reader/useReadingPosition'
import {
  useReaderSettings, directionFor, type Direction,
  type SourcePreference,
} from '@features/reader/store'
import { useWorkData } from '@features/work/useWorkData'
import { useChapterSpawn } from '@features/work/WorkChapterList'
import type { HubChapter, HubVersion } from '@features/title/mergeChapters'


interface SearchParams {
  page?: number
}


function ReaderPage() {
  const { workId: workIdStr, numberNorm } = Route.useParams()
  const { page = 0 } = Route.useSearch()
  const nav = Route.useNavigate()
  const workId = Number(workIdStr)
  const validWorkId = Number.isInteger(workId) && workId > 0

  const setPage = useCallback((p: number) =>
    nav({ search: (s) => ({ ...s, page: p > 0 ? p : undefined }) }),
  [nav])

  const reader = useReader({
    workId:     validWorkId ? workId : 0,
    numberNorm,
  })

  const setSourcePref = useReaderSettings((s) => s.setSourcePref)

  // Direction = source of truth for body layout. Per-work setting.
  const direction: Direction = useReaderSettings(
    (s) => directionFor(s, workId),
  )
  const isTTB = direction === 'ttb'

  const ui = useReaderUiState()

  // Trigger refs for the top-bar dropdowns. Both popovers anchor
  // against their trigger button so click-outside detection is
  // unambiguous (the trigger is part of the "inside" set).
  const chapterTriggerRef = useRef<HTMLButtonElement>(null)
  const sourceTriggerRef  = useRef<HTMLButtonElement>(null)

  // Visible page tracker — drives the bottom-pill chapter X/Y. In
  // pager mode it mirrors `page`; in TTB it tracks scroll via the
  // StripView's IntersectionObserver callback.
  const [visiblePage, setVisiblePage] = useState(0)
  useEffect(() => {
    if (!isTTB) setVisiblePage(page)
  }, [page, isTTB])

  // Single-mode past-end: user clicked next on the last page; show
  // end-of-chapter card instead of stuck-on-last-page UX.
  const [pastEnd, setPastEnd] = useState(false)
  useEffect(() => { setPastEnd(false) }, [workIdStr, numberNorm, direction])

  // Resume position — local storage only at this stage; cross-device
  // sync goes through server `record*Reading` already (chapter-level)
  // and a future endpoint for intra-chapter position.
  const totalPages = reader.pages.length
  const resumeEnabled = useReaderSettings((s) => s.resumePosition)
  const persist = useReadingPosition(workId, numberNorm, totalPages)

  // Guard: restore should run AT MOST ONCE per (chapter, mode). The
  // previous version keyed an effect on `totalPages > 0`, which
  // flips back to false transiently when the bunle is between
  // chapters \u2014 then back to true \u2014 firing restore again and
  // yanking the user back to the saved position (often 0%) mid-scroll.
  // The ref tracks "did I already restore for this chapter slug"
  // and is reset by the chapter-change effect below.
  const restoredRef = useRef<string | null>(null)
  useEffect(() => { restoredRef.current = null }, [workIdStr, numberNorm, isTTB])

  useEffect(() => {
    if (!validWorkId || totalPages <= 0) return
    const key = `${workIdStr}::${numberNorm}::${isTTB ? 'ttb' : 'pager'}`
    if (restoredRef.current === key) return
    restoredRef.current = key

    if (!resumeEnabled) {
      window.scrollTo({ top: 0, behavior: 'auto' })
      return
    }
    const saved = loadPosition(workId, numberNorm)
    if (!saved) {
      window.scrollTo({ top: 0, behavior: 'auto' })
      return
    }
    if (!isTTB) {
      const target = Math.min(saved.page, totalPages - 1)
      if (target > 0 && target !== page) {
        nav({ search: (s) => ({ ...s, page: target }), replace: true })
      }
      window.scrollTo({ top: 0, behavior: 'auto' })
      return
    }
    // TTB: wait a frame so the strip's first-paint heights land
    // before measuring scrollHeight. One-shot \u2014 we never call
    // this again for this chapter.
    requestAnimationFrame(() => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      window.scrollTo({
        top: Math.max(0, max * saved.scrollPct),
        behavior: 'auto',
      })
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps -- gated by restoredRef
  }, [workIdStr, numberNorm, totalPages > 0, isTTB, resumeEnabled])

  // Persist on scroll only. The previous code ALSO wrote on every
  // `visiblePage` change with `scrollPct: 0`, which raced with the
  // scroll listener: a write of pct=0 would land in localStorage
  // and the next chapter open / re-render would yank the user back
  // to the top.
  useEffect(() => {
    if (totalPages <= 0 || !isTTB || !resumeEnabled) return
    let raf: number | null = null
    const onScroll = () => {
      if (raf !== null) return
      raf = requestAnimationFrame(() => {
        const max = document.documentElement.scrollHeight - window.innerHeight
        const pct = max > 0 ? window.scrollY / max : 0
        persist.update({ page: visiblePage, scrollPct: pct })
        raf = null
      })
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [isTTB, persist, visiblePage, totalPages, resumeEnabled])

  // Pager mode persists per-page change (no scroll), so we keep a
  // separate writer here keyed on `page` rather than the scroll
  // listener. `scrollPct: 0` is correct in pager mode \u2014 there's
  // no vertical scroll within a chapter.
  useEffect(() => {
    if (totalPages <= 0 || isTTB || !resumeEnabled) return
    persist.update({ page, scrollPct: 0 })
  }, [page, isTTB, totalPages, persist, resumeEnabled])

  // Spawn pipeline — for end-of-chapter "Dịch từ X" CTA. Re-uses
  // the work-list hook so progress state surfaces consistently
  // across hub and reader.
  const work = useWorkData(validWorkId ? workId : 0)
  const spawnCtl = useChapterSpawn(work.targetLang, validWorkId ? workId : undefined)

  // Cache "what was the spawn target?" per version key. When the
  // server-side translation finishes rendering (state flips to
  // 'done' on the next work-payload refetch), the watcher below
  // pulls the (chapter, raw) pair out of this map to build the
  // toast label + the click-to-read action.
  //
  // Stored as a ref so the watcher effect's identity doesn't
  // recreate on every spawn. Entries are NOT removed after the
  // toast fires — the user might dismiss and re-spawn later; the
  // next spawn's "done" only counts the freshly-pending row, never
  // a leftover entry, because we key on (number, source_lang) and
  // each spawn writes to that key.
  const spawnTargetsRef = useRef<Map<string, {
    chapter: HubChapter
    raw:     HubVersion
  }>>(new Map())

  // Track which translations we've already announced. Keys are
  // `${chapter.number}::${source_lang}` so a spawn from EN and a
  // spawn from KR on the same chapter each get their own toast.
  const announcedRef = useRef<Set<string>>(new Set())

  const handleSpawnNext = useCallback(
    (chapter: HubChapter, raw: HubVersion) => {
      const key = `${chapter.number}::${normalizeBcp(raw.lang)}`
      spawnTargetsRef.current.set(key, { chapter, raw })
      // Allow re-announcing: user kicked a fresh spawn for this
      // (chapter, source) pair. The previous toast may have been
      // dismissed; we want the new "done" to surface.
      announcedRef.current.delete(key)
      spawnCtl.spawn(chapter, raw)
    },
    [spawnCtl],
  )

  // Watcher — fires a toast when the SERVER-SIDE translation
  // state actually flips to 'done' (i.e. the worker rendered the
  // archive), not when the client-side spawn pipeline merely
  // finished POSTing. The previous version watched
  // `spawnCtl.progressByKey[*].phase === 'done'`, but that phase
  // means "I uploaded + posted /translate" — at that point the
  // server has only a `pending` draft, the bunle archive doesn't
  // exist yet, and the user reading the toast and clicking "Đọc"
  // would land on a `pending-render` page.
  //
  // We scan work.chapters for translations whose state is 'done'
  // and whose (chapter, source_lang) key is in our pending-spawn
  // map but not yet announced. Triggered every time the work
  // payload refetches (`work.chapters` changes identity).
  useEffect(() => {
    if (work.chapters.length === 0) return
    for (const [key, target] of spawnTargetsRef.current.entries()) {
      if (announcedRef.current.has(key)) continue
      const wantSrc = normalizeBcp(target.raw.lang)
      const chapter = work.chapters.find(
        (c) => c.number === target.chapter.number,
      )
      if (!chapter) continue
      const tx = chapter.versions.find(
        (v) => v.kind === 'translation'
            && v.state === 'done'
            && normalizeBcp(v.sourceLang) === wantSrc,
      )
      if (!tx) continue

      announcedRef.current.add(key)

      const srcLabel = target.raw.lang
        ? target.raw.lang.toUpperCase()
        : 'nguồn'
      toast.success(
        `Đã dịch xong Ch.${chapter.number} từ ${srcLabel}`,
        {
          action: {
            label: 'Đọc',
            onClick: () => {
              // The user just consented to read this version. Pin
              // pref so future chapters auto-pick the matching
              // (target_lang, source_lang) draft — same contract
              // as tapping a row in the source picker.
              if (work.targetLang) {
                setSourcePref(workId, {
                  kind:       'translation',
                  lang:       tx.lang,
                  sourceLang: tx.sourceLang ?? null,
                })
              }
              if (chapter.number !== numberNorm) {
                nav({
                  to: '/r/$workId/$numberNorm',
                  params: {
                    workId:     String(workId),
                    numberNorm: chapter.number,
                  },
                })
              }
            },
          },
        },
      )
    }
  }, [
    work.chapters, work.targetLang,
    setSourcePref, workId, numberNorm, nav,
  ])

  // Source picker handler. Picking a row commits the user's choice
  // immediately: pref is written for the work, so the current
  // chapter swaps to the new version AND prev/next chapters
  // inherit the same (kind, lang, sourceLang) selection without
  // the user having to re-pick. No "save as default" toggle —
  // every tap IS the commitment.
  const handlePickSource = useCallback(
    (version: HubVersion) => {
      if (!work.targetLang) return
      const pref: SourcePreference = version.kind === 'translation'
        ? {
            kind:       'translation',
            lang:       version.lang,
            sourceLang: version.sourceLang ?? null,
          }
        : {
            kind: 'raw',
            lang: version.lang,
          }
      setSourcePref(workId, pref)
    },
    [workId, work.targetLang, setSourcePref],
  )

  const readingSourceLang = useMemo(() => {
    const p = reader.picked
    if (!p) return null
    if (p.kind === 'translation') return p.sourceLang ?? null
    return p.lang
  }, [reader.picked])

  const onBackWork = useCallback(() => nav({
    to: '/w/$workId', params: { workId: workIdStr },
  }), [nav, workIdStr])

  // Chapter index 1-based for "Ch.X / Y" display.
  const chapterIndex = useMemo(() => {
    if (!reader.chapter) return 0
    const idx = work.chapters.findIndex(
      (c) => c.number === reader.chapter!.number,
    )
    if (idx < 0) return 0
    // Spine sorted DESC (latest first). Position 1 should mean the
    // oldest chapter so the count reads naturally ("Ch.1 / 58"
    // matches index 1). Invert via length - idx.
    return work.chapters.length - idx
  }, [reader.chapter, work.chapters])

  // Tap-zone callbacks. Left/right semantics:
  //   - TTB:         always prev/next chapter (vertical scroll
  //                  handles intra-chapter; horizontal tap = jump).
  //   - Pager:       prev/next page; at edge, jump chapter.
  const onTapPrev = useCallback(() => {
    if (!isTTB && page > 0) { setPage(page - 1); return }
    const t = reader.nav.prev
    if (t) nav({
      to: '/r/$workId/$numberNorm',
      params: { workId: String(t.workId), numberNorm: t.numberNorm },
    })
  }, [isTTB, page, setPage, reader.nav.prev, nav])

  const onTapNext = useCallback(() => {
    if (!isTTB) {
      if (page < totalPages - 1) { setPage(page + 1); return }
      if (page >= totalPages - 1) { setPastEnd(true); return }
    }
    const t = reader.nav.next
    if (t) nav({
      to: '/r/$workId/$numberNorm',
      params: { workId: String(t.workId), numberNorm: t.numberNorm },
    })
  }, [isTTB, page, totalPages, setPage, reader.nav.next, nav])

  const onPrevChapter = useCallback(() => {
    const t = reader.nav.prev
    if (t) nav({
      to: '/r/$workId/$numberNorm',
      params: { workId: String(t.workId), numberNorm: t.numberNorm },
    })
  }, [reader.nav.prev, nav])

  const onNextChapter = useCallback(() => {
    const t = reader.nav.next
    if (t) nav({
      to: '/r/$workId/$numberNorm',
      params: { workId: String(t.workId), numberNorm: t.numberNorm },
    })
  }, [reader.nav.next, nav])

  useReaderKeyboard({
    onPrev:         onTapPrev,
    onNext:         onTapNext,
    onPrevChapter,
    onNextChapter,
    onToggleChrome:    ui.toggleChrome,
    onOpenSettings:    ui.openSettings,
    onOpenChapterList: ui.openChapterList,
  })

  if (!validWorkId) {
    return (
      <div className="px-4 py-16">
        <EmptyState icon={AlertTriangle} title="Đường dẫn không hợp lệ" />
      </div>
    )
  }

  const endSlot = (
    <EndOfChapterCard
      workId={workId}
      currentLabel={reader.meta.chapterText}
      readingSourceLang={readingSourceLang}
      targetLang={work.targetLang}
      chapters={work.chapters}
      currentNum={numberNorm}
      onSpawn={handleSpawnNext}
      onBackToList={onBackWork}
    />
  )

  return (
    <div className="min-h-dvh bg-bg">
      <ReaderTopBar
        hidden={ui.chromeHidden}
        workTitle={reader.meta.workTitle || 'Chương'}
        chapterLabel={reader.meta.chapterText}
        totalChapters={work.chapters.length}
        chapterIndex={chapterIndex}
        picked={reader.picked}
        chapterTriggerRef={chapterTriggerRef}
        sourceTriggerRef={sourceTriggerRef}
        onBackWork={onBackWork}
        onOpenChapters={ui.openChapterList}
        onOpenSources={ui.openSources}
      />

      {/* Top spacer so the body doesn't slide under the fixed
          top bar. Bottom is handled by the pill being floating
          rather than reserving layout space. */}
      <div className="pt-[calc(var(--sait)+var(--spacing-bar))]">
        {reader.prefMismatch && (
          <PrefMismatchBanner
            onOpenSources={ui.openSources}
          />
        )}
        <ReaderContent
          reader={reader}
          numberNorm={numberNorm}
          page={page}
          direction={direction}
          endSlot={endSlot}
          pastEnd={pastEnd}
          inputDisabled={ui.anySheetOpen}
          onVisiblePageChange={setVisiblePage}
          onTapPrev={onTapPrev}
          onTapNext={onTapNext}
          onTogglePeek={ui.toggleChrome}
          onPastEnd={() => setPastEnd(true)}
        />
      </div>

      <ReaderBottomPill
        hidden={ui.chromeHidden}
        prev={reader.nav.prev}
        next={reader.nav.next}
        onOpenSettings={ui.openSettings}
      />

      <SettingsSheet
        open={ui.settingsOpen}
        onClose={ui.closeSettings}
        workId={workId}
      />

      <ChapterPicker
        open={ui.chapterListOpen}
        onClose={ui.closeChapterList}
        anchorRef={chapterTriggerRef}
        workId={workId}
        chapters={work.chapters}
        currentNum={numberNorm}
        targetLang={work.targetLang}
      />

      <SourcePicker
        open={ui.sourcesOpen}
        onClose={ui.closeSources}
        anchorRef={sourceTriggerRef}
        targetLang={work.targetLang}
        chapter={reader.chapter}
        picked={reader.picked}
        spawnState={
          reader.chapter
            ? spawnCtl.getSpawnState(reader.chapter.number)
            : null
        }
        onPick={handlePickSource}
        onSpawn={handleSpawnNext}
      />
    </div>
  )
}


/** Renders the body — pages OR a status placeholder. Both top bar
 *  and bottom pill stay mounted above regardless of status so the
 *  user can navigate sibling chapters / open settings while the
 *  current chapter is loading or errored. */
function ReaderContent({
  reader, numberNorm, page, direction, endSlot, pastEnd,
  inputDisabled, onVisiblePageChange,
  onTapPrev, onTapNext, onTogglePeek, onPastEnd,
}: {
  reader:        ReturnType<typeof useReader>
  numberNorm:    string
  page:          number
  direction:     Direction
  endSlot:       React.ReactNode
  pastEnd:       boolean
  inputDisabled: boolean
  onVisiblePageChange: (idx: number) => void
  onTapPrev:    () => void
  onTapNext:    () => void
  onTogglePeek: () => void
  onPastEnd:    () => void
}) {
  if (reader.status === 'loading') {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (reader.status === 'not-found') {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Không tìm thấy chương"
          hint={`Chương "${numberNorm}" không có trong manga này.`}
        />
      </div>
    )
  }
  if (reader.status === 'empty') {
    return <div className="pt-4">{endSlot}</div>
  }
  if (reader.status === 'pending-render') {
    return (
      <div className="px-6 py-16 text-center max-w-md mx-auto">
        <p className="text-sm font-medium text-text">Chương chưa render xong</p>
        <p className="text-xs text-text-subtle mt-1">
          Bản dịch đang được hệ thống tạo ra — quay lại sau ít phút.
        </p>
      </div>
    )
  }
  if (reader.status === 'no-source') {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Nguồn chưa được cài"
          hint="Vào Cài đặt → Nguồn để cài plugin tương ứng."
        />
      </div>
    )
  }
  if (reader.status === 'error') {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được chương"
          hint={reader.error ?? 'Selector hoặc archive có thể đã hỏng.'}
        />
      </div>
    )
  }
  return (
    <ReaderBody
      source={reader}
      direction={direction}
      page={page}
      pastEnd={pastEnd}
      endSlot={endSlot}
      inputDisabled={inputDisabled}
      onVisiblePageChange={onVisiblePageChange}
      onPrev={onTapPrev}
      onNext={onTapNext}
      onTogglePeek={onTogglePeek}
      onPastEnd={onPastEnd}
    />
  )
}


export const Route = createFileRoute('/r/$workId/$numberNorm')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    page: typeof s.page === 'number' && s.page > 0 ? s.page : undefined,
  }),
  beforeLoad: async ({ params, search, context }) => {
    const workId = Number(params.workId)
    if (!Number.isInteger(workId) || workId <= 0) return
    try {
      await context.queryClient.ensureQueryData({
        queryKey: qk.work.byId(workId),
        queryFn:  () => api.getWork(workId),
      })
    } catch (err) {
      if (err instanceof WorkRedirectedError) {
        throw routerRedirect({
          to:     '/r/$workId/$numberNorm',
          params: {
            workId:     String(err.newId),
            numberNorm: params.numberNorm,
          },
          search,
          replace: true,
        })
      }
    }
  },
  component: ReaderPage,
  staticData: { chrome: 'bare' },
})


/** Inline banner shown when the user's saved source pref doesn't
 *  match anything on the current chapter. Lets them open the
 *  source picker to switch to whatever IS available (or trigger a
 *  Dịch spawn). Picking a new row rewrites the pref, so the
 *  banner naturally disappears — we don't expose a separate
 *  "clear pref" action because there is no useful state to clear
 *  TO: the only way to have no pref is to never have picked, and
 *  re-picking is one tap away. */
function PrefMismatchBanner({
  onOpenSources,
}: {
  onOpenSources: () => void
}) {
  return (
    <div className="px-4 sm:px-6 pt-3">
      <div className="rounded-md bg-info-bg text-info-text border border-info/30 px-4 py-3 flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">
            Chương này không có nguồn bạn đang đọc
          </p>
          <p className="text-xs opacity-80 mt-0.5">
            Đang hiển thị bản thay thế. Chọn nguồn khác hoặc dịch ngay.
          </p>
        </div>
        <div className="shrink-0">
          <button
            onClick={onOpenSources}
            className="text-xs font-medium px-3 h-7 rounded-md bg-info text-bg hover:opacity-90 transition-opacity cursor-pointer"
          >
            Chọn nguồn
          </button>
        </div>
      </div>
    </div>
  )
}
