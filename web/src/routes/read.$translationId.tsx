// Chapter reader route — /read/$translationId
//
// Owns its own chrome (bare shell). The toolbar at the top handles
// breadcrumb + mode + prev/next; the body streams the BNL archive and
// renders pages either continuously or one at a time.
//
// Prev/next: we fetch the material the translation lives in, find the
// current chapter's position, then look for adjacent chapters that
// already have a `done` translation in the same target_lang. Chapters
// without a finished translation are skipped — clicking next jumps to
// the next translated chapter, not the next raw chapter.

import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useEffect, useMemo, useRef } from 'react'
import { AlertTriangle } from 'lucide-react'
import { api, type ApiChapter } from '@shared/api/api'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { ReaderToolbar, type ViewMode } from '@features/reader/ReaderToolbar'
import { ContinuousView } from '@features/reader/ContinuousView'
import { SinglePageView } from '@features/reader/SinglePageView'
import { useChapterArchive } from '@features/reader/useChapterArchive'
import {
  clearReadingPresence, lockPortrait, setReadingPresence, unlockOrientation,
} from '@shared/discord/presence'

interface SearchParams {
  page?: number
  mode?: ViewMode
}

function ReaderPage() {
  const { translationId } = Route.useParams()
  const { page = 0, mode = 'continuous' } = Route.useSearch()
  const nav = Route.useNavigate()
  const tid = Number(translationId)
  const valid = Number.isInteger(tid) && tid > 0

  const setPage = (p: number) =>
    nav({ search: (s) => ({ ...s, page: p > 0 ? p : undefined }) })
  const setMode = (m: ViewMode) =>
    nav({ search: (s) => ({ ...s, mode: m === 'continuous' ? undefined : m }) })

  // Translation row — gives us archive_url + chapter + material context.
  const { data: trans, isPending, isError, error } = useQuery({
    queryKey: ['translation', tid],
    queryFn:  () => api.getTranslation(tid),
    enabled:  valid,
  })

  // Material — used to locate prev/next chapters with done translations
  // in the same target_lang. Cached for 60s; readers rarely jump beyond
  // a chapter and the chapter list is already cached by the hub.
  const { data: material } = useQuery({
    queryKey: ['material', 'detail', trans?.material_id],
    queryFn:  () => api.getMaterial(trans!.material_id),
    enabled:  !!trans?.material_id,
    staleTime: 60_000,
  })

  // Adjacent done translations in the same target_lang. Returns the
  // translation_id you'd navigate to, not the chapter_id.
  const { prevTransId, nextTransId } = useMemo(() => {
    if (!material || !trans) return { prevTransId: null, nextTransId: null }
    const chapters = [...material.chapters].sort(
      (a: ApiChapter, b: ApiChapter) => a.position - b.position,
    )
    const idx = chapters.findIndex((c) => c.id === trans.chapter_id)
    if (idx < 0) return { prevTransId: null, nextTransId: null }

    const findDone = (start: number, step: -1 | 1): number | null => {
      for (let i = start; i >= 0 && i < chapters.length; i += step) {
        const ch = chapters[i]!
        const done = ch.translations.find(
          (t) => t.target_lang === trans.target_lang && t.state === 'done',
        )
        if (done) return done.id
      }
      return null
    }
    return {
      prevTransId: findDone(idx - 1, -1),
      nextTransId: findDone(idx + 1, +1),
    }
  }, [material, trans])

  const { bunle, urls, loading: aLoading, error: aError } =
    useChapterArchive(trans?.archive_url)

  // Reading history — fire-and-forget when the chapter resolves.
  // Records (user, chapter, translation) so the home page can build
  // a "Tiếp tục đọc" surface without depending on bookmarks. UPSERT
  // on the server side; replaying the same chapter is idempotent.
  // Ref guards against duplicate POSTs from React strict-mode double
  // mount and from any future query refetch.
  const recordedRef = useRef<number | null>(null)
  const recordReading = useMutation({
    mutationFn: api.recordTranslatedReading,
  })
  useEffect(() => {
    if (!trans) return
    if (recordedRef.current === trans.id) return
    recordedRef.current = trans.id
    recordReading.mutate({ translation_id: trans.id })
  // eslint-disable-next-line react-hooks/exhaustive-deps -- recordReading is stable
  }, [trans?.id])

  // Discord Activity presence + portrait lock — same lifecycle as the
  // legacy reader: presence follows the current chapter, lock toggles
  // once per mount.
  const materialTitle = trans?.material_title ?? ''
  const chapterNumber = trans?.chapter_number ?? ''
  const chapterLabel  = trans?.chapter_label ?? null
  useEffect(() => {
    if (!materialTitle || !chapterNumber) return
    setReadingPresence({
      projectTitle:  materialTitle,
      chapterNumber: chapterNumber,
      chapterTitle:  chapterLabel,
    })
  }, [materialTitle, chapterNumber, chapterLabel])
  useEffect(() => {
    lockPortrait()
    return () => {
      unlockOrientation()
      clearReadingPresence()
    }
  }, [])

  if (!valid) {
    return (
      <div className="px-4 py-16">
        <EmptyState icon={AlertTriangle} title="Translation không hợp lệ" />
      </div>
    )
  }

  if (isPending) {
    return (
      <div className="flex items-center justify-center min-h-dvh">
        <Spinner size={20} />
      </div>
    )
  }

  if (isError || !trans) {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được translation"
          hint={(error as Error)?.message ?? 'Có thể đã bị xóa.'}
        />
      </div>
    )
  }

  if (!trans.archive_url) {
    return (
      <div className="px-6 py-16 text-center max-w-md mx-auto">
        <p className="text-sm font-medium text-text">Chương chưa render xong</p>
        <p className="text-xs text-text-subtle mt-1">
          Trạng thái: {trans.state}. Khi worker render xong, mở lại trang này.
        </p>
      </div>
    )
  }

  return (
    <div className="min-h-dvh bg-bg">
      <ReaderToolbar
        entryTitle={materialTitle || 'Chương'}
        chapterNumber={chapterNumber}
        chapterLabel={chapterLabel}
        prevTransId={prevTransId}
        nextTransId={nextTransId}
        page={page}
        totalPages={bunle?.pageCount ?? 0}
        mode={mode}
        onModeChange={setMode}
        onBack={() => window.history.back()}
      />

      {aLoading && (
        <div className="flex items-center justify-center py-24 text-text-subtle">
          <Spinner size={20} />
        </div>
      )}

      {aError && (
        <div className="py-16 text-center">
          <p className="text-sm text-error-text font-medium">Không tải được archive</p>
          <p className="text-xs text-text-subtle mt-1">{aError}</p>
        </div>
      )}

      {bunle && mode === 'continuous' && (
        <ContinuousView bunle={bunle} urls={urls} />
      )}

      {bunle && mode === 'single' && (
        <SinglePageView
          bunle={bunle}
          urls={urls}
          page={page}
          onChange={setPage}
        />
      )}
    </div>
  )
}

export const Route = createFileRoute('/read/$translationId')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    page: typeof s.page === 'number' && s.page > 0 ? s.page : undefined,
    mode: s.mode === 'single' ? 'single' : undefined,
  }),
  component: ReaderPage,
  staticData: { chrome: 'bare' },
})
