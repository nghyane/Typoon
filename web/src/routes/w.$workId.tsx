// /w/$workId — canonical Work-centric manga page.
//
// One Work groups N sibling materials. The page renders:
//   • Hero (active material's metadata + sibling chip rail)
//   • Continue-reading bar (server-side reading_history latest hit)
//   • Chapter list (manifest spine of the active source overlaid
//     with cross-source translation rows)
//
// URL state:
//   /w/123        → defaults to oldest installed-source material
//   /w/123?src=7  → forces material 7 as the active source
//
// Per-Work metadata is sparse (Cách 1 — danh bạ); display fields come
// from the active material. Switching the source re-fetches its
// manifest live; the cross-source translation overlay is already
// part of the work payload.

import { useCallback, useMemo } from 'react'
import {
  createFileRoute, useNavigate, redirect as routerRedirect,
} from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle } from 'lucide-react'

import { api, WorkRedirectedError } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'

import { useWorkData } from '@features/work/useWorkData'
import { useAutoEnrichWork } from '@features/link/useAutoEnrichWork'
import {
  WorkHero, ContinueReadingBar,
} from '@features/work/WorkHero'
import { LinkSuggestionPanel } from '@features/work/LinkSuggestionPanel'
import {
  WorkChapterList, useChapterSpawn,
} from '@features/work/WorkChapterList'
import {
  type HubChapter, type HubVersion,
} from '@features/title/mergeChapters'


interface SearchParams {
  /** Active source — sibling material id. Absent → server default
   *  (oldest installed-source material). */
  src?: number
}


function WorkPage() {
  const { workId } = Route.useParams()
  const { src }    = Route.useSearch()
  const nav        = useNavigate({ from: '/w/$workId' })

  const workIdNum = Number(workId)

  const {
    work, materials, activeMaterial, targetLang, chapters,
    workLoading, manifestLoading, workError,
  } = useWorkData(workIdNum, src ?? null)

  // Silent cross-reference auto-enrich. When this Work has no
  // `cross_refs` yet, the hook fans search across installed link
  // plugins (Anilist, …), POSTs whatever it found back to the
  // server, and the linker takes it from there. Fires at most once
  // per (work, week); no UI surface for the user.
  useAutoEnrichWork(work)

  // Recent-read row scoped to this Work — drives the "Tiếp tục"
  // affordance in the hero. Pulled out of the global recent feed by
  // workId match.
  const recentQ = useQuery({
    queryKey: qk.me.recentReads(),
    queryFn:  () => api.listRecentReads(50),
    staleTime: 30_000,
  })
  const resumeFrom = useMemo(
    () => (recentQ.data ?? []).find((r) => r.work_id === workIdNum) ?? null,
    [recentQ.data, workIdNum],
  )

  const handleSelectSource = useCallback((materialId: number) => {
    nav({
      params: { workId },
      search: { src: materialId },
    })
  }, [nav, workId])

  const handleResume = useCallback(() => {
    if (!resumeFrom) return
    // Unified reader URL — same path covers translation + raw resume.
    // `useReader` picks the right kind from the cached Work payload.
    nav({
      to:     '/r/$workId/$numberNorm',
      params: {
        workId:     String(workIdNum),
        numberNorm: resumeFrom.chapter_number,
      },
      search: {
        lang: targetLang ?? undefined,
        src:  activeMaterial?.id,
      },
    })
  }, [resumeFrom, nav, workIdNum, targetLang, activeMaterial])

  const handleShare = useCallback(async () => {
    const url = `${window.location.origin}/w/${workId}`
    if (navigator.share) {
      try {
        await navigator.share({
          title: activeMaterial?.title ?? 'Manga',
          url,
        })
      } catch {
        /* dismissed */
      }
    } else {
      void navigator.clipboard?.writeText(url)
    }
  }, [workId, activeMaterial])

  // Spawn — wraps useSpawnChapter so only the active row reflects
  // progress text. Spawn is per-raw-version: the user picks which
  // source language they want the LLM to translate from.
  const spawnCtl = useChapterSpawn(targetLang)
  const handleSpawn = useCallback(
    (chapter: HubChapter, raw: HubVersion) => {
      spawnCtl.spawn(raw, chapter.label)
    },
    [spawnCtl],
  )

  // Retry — separate path from spawn. Failed translation rows have
  // no upstream/material handles (only `translationId`), so the
  // upload-and-spawn pipeline would silently no-op. `redo` reuses the
  // server-side chapter bytes and re-runs the LLM stages.
  const qc = useQueryClient()
  const redoMut = useMutation({
    mutationFn: (translationId: number) => api.redoTranslation(translationId),
    onSuccess:  () => { void qc.invalidateQueries({ queryKey: qk.work.all() }) },
  })
  const handleRetryTranslation = useCallback(
    (translationId: number) => {
      if (redoMut.isPending) return
      redoMut.mutate(translationId)
    },
    [redoMut],
  )

  const handleOpenVersion = useCallback((c: HubChapter) => {
    // Unified reader URL — picks translation vs raw client-side from
    // the cached Work payload. URL stays stable across kind changes,
    // so a deep link to ch.64 reads the translated version once it
    // exists without needing to share a different URL.
    nav({
      to:     '/r/$workId/$numberNorm',
      params: { workId: String(workIdNum), numberNorm: c.number },
      search: {
        lang: targetLang ?? undefined,
        src:  activeMaterial?.id,
      },
    })
  }, [nav, workIdNum, targetLang, activeMaterial])

  // ── Render ─────────────────────────────────────────────────────

  if (workLoading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (workError || !work) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được"
          hint={workError?.message ?? 'Manga không tồn tại hoặc đã bị xoá.'}
        />
      </div>
    )
  }

  const latestChapter = chapters[0]?.number ?? null

  return (
    <div className="pb-16">
      <WorkHero
        workId={workIdNum}
        activeMaterial={activeMaterial}
        materials={materials}
        resumeFrom={resumeFrom}
        viewerEntry={work.viewer_entry}
        latestChapterNum={latestChapter}
        totalChapters={chapters.length}
        onSelectSource={handleSelectSource}
        onShare={handleShare}
        onResume={handleResume}
      />

      {/* Optional resume bar shown below the hero on touch widths */}
      <div className="px-4 sm:px-6 mb-3 sm:hidden">
        <ContinueReadingBar
          resumeFrom={resumeFrom}
          onResume={handleResume}
        />
      </div>

      <LinkSuggestionPanel workId={workIdNum} ownMaterials={materials} />

      <WorkChapterList
        chapters={chapters}
        targetLang={targetLang}
        loading={manifestLoading}
        spawnState={spawnCtl.progress}
        spawningKey={spawnCtl.spawningKey}
        onSpawn={handleSpawn}
        onRetryTranslation={handleRetryTranslation}
        onOpenVersion={handleOpenVersion}
      />
    </div>
  )
}


export const Route = createFileRoute('/w/$workId')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    src: typeof s.src === 'number' ? s.src : undefined,
  }),
  // Intercept the Work merge redirect BEFORE rendering the page. If
  // the URL points at a dissolved Work, ensure-fetch surfaces
  // `WorkRedirectedError` carrying the canonical id; we throw a
  // router `redirect()` so the navigation happens at the routing
  // layer (no flash, no useEffect, no cache pollution under the
  // stale key).
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
          to:      '/w/$workId',
          params:  { workId: String(err.newId) },
          search,
          replace: true,
        })
      }
      // Other errors propagate to the component, which renders the
      // existing error UI from `workError`.
    }
  },
  component: WorkPage,
})
