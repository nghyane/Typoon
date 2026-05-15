// /w/$workId — canonical Work-centric manga page.
//
// One Work groups N sibling materials. The page renders:
//   • Hero (resolved primary-material metadata, target-lang biased)
//   • Continue-reading bar (server-side reading_history latest hit)
//   • Chapter list (union of every installed source's manifest spine
//     overlaid with cross-source translation rows)
//
// No URL state for source selection — chapters auto-merge across
// every installed-source material attached to the Work. Each row's
// `ChapterRow` already shows the per-chapter source, so a chip rail
// above the list would be redundant.

import { useCallback, useMemo, useState } from 'react'
import {
  createFileRoute, useNavigate, redirect as routerRedirect,
} from '@tanstack/react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle } from 'lucide-react'

import { api, WorkRedirectedError } from '@shared/api/api'
import { qk } from '@shared/api/keys'
import { toast } from '@shared/ui/Toaster'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'

import { useWorkData } from '@features/work/useWorkData'
import { useAutoEnrichWork } from '@features/link/useAutoEnrichWork'
import {
  WorkHero, ContinueReadingBar,
} from '@features/work/WorkHero'
import { LinkSuggestionPanel } from '@features/work/LinkSuggestionPanel'
import { WorkMembersPanel } from '@features/work/WorkMembersPanel'
import { UploadChapterDialog } from '@features/work/UploadChapterDialog'
import {
  WorkChapterList, useChapterSpawn,
} from '@features/work/WorkChapterList'
import { pickPrimaryMaterial, resolveWorkTitle } from '@features/work/title'
import {
  type HubChapter, type HubVersion,
} from '@features/title/mergeChapters'


function WorkPage() {
  const { workId } = Route.useParams()
  const nav        = useNavigate({ from: '/w/$workId' })

  const workIdNum = Number(workId)

  const {
    work, materials, targetLang, chapters,
    workLoading, manifestsLoading, workError,
  } = useWorkData(workIdNum)

  // Single primary material drives share-title / fallback labels.
  // Same resolver as the hero so both pick the same handle.
  const primary = useMemo(
    () => pickPrimaryMaterial(materials, targetLang),
    [materials, targetLang],
  )

  // Silent cross-reference auto-enrich. When this Work has no
  // `cross_refs` yet, the hook fans search across installed link
  // plugins (Anilist, …), POSTs whatever it found back to the
  // server, and the linker takes it from there. Fires at most once
  // per (work, week); no UI surface for the user.
  useAutoEnrichWork(work, targetLang)

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
    })
  }, [resumeFrom, nav, workIdNum])

  const handleShare = useCallback(async () => {
    const url = `${window.location.origin}/w/${workId}`
    if (navigator.share) {
      try {
        await navigator.share({
          title: primary?.title ?? 'Manga',
          url,
        })
      } catch {
        /* dismissed */
      }
    } else {
      void navigator.clipboard?.writeText(url)
    }
  }, [workId, primary])

  // Upload-chapter dialog: state lives at the route so both the hero
  // action button and a future empty-state CTA on a fresh "Tạo trống"
  // work can mount the same dialog without prop-drilling.
  const [uploadOpen, setUploadOpen] = useState(false)
  const existingChapterNumbers = useMemo(
    () => new Set(chapters.map((c) => c.number)),
    [chapters],
  )
  const workTitleForUpload = useMemo(
    () => resolveWorkTitle(materials, targetLang).title,
    [materials, targetLang],
  )

  // Spawn — chapter-keyed pipeline. One slot per chapter so the same
  // row tracks raw → upload → server-pending → done without ever
  // moving in the list.
  const spawnCtl = useChapterSpawn(targetLang, workIdNum)
  const handleSpawn = useCallback(
    (chapter: HubChapter, raw: HubVersion) => {
      spawnCtl.spawn(chapter, raw)
    },
    [spawnCtl],
  )
  const handleAbort = useCallback(
    (chapter: HubChapter) => spawnCtl.abort(chapter),
    [spawnCtl],
  )

  // Retry — separate path from spawn. Failed translation rows have
  // no upstream/material handles (only `translationId`), so the
  // upload-and-spawn pipeline would silently no-op. `redo` reuses the
  // server-side chapter bytes and re-runs the LLM stages.
  //
  // Server semantics (see typoon/api/routes/translate.py:redo_translation):
  //   - error              → restart + state='pending', cache_hit=false
  //   - done               → no-op, state='done', cache_hit=true
  //   - pending/running    → no-op, returns live state, cache_hit=true
  //   - blocked            → 409
  // We surface a different toast per outcome so the user knows
  // whether the click actually re-kicked the pipeline or just
  // confirmed an already-running one.
  const qc = useQueryClient()
  const redoMut = useMutation({
    mutationFn: (translationId: number) => api.redoTranslation(translationId),
    onSuccess:  (res) => {
      if (res.cache_hit && res.state === 'done') {
        toast.success('Bản dịch đã hoàn tất, không cần thử lại.')
      } else if (res.cache_hit) {
        toast.success('Pipeline đang chạy — sẽ tự cập nhật khi xong.')
      } else {
        toast.success('Đã khởi động lại — đang dịch.')
      }
      void qc.invalidateQueries({ queryKey: qk.work.all() })
    },
    onError: (e: Error) => toast.error(`Thử lại thất bại: ${e.message}`),
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
    })
  }, [nav, workIdNum])

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
        materials={materials}
        resumeFrom={resumeFrom}
        viewerEntry={work.viewer_entry}
        targetLang={targetLang}
        latestChapterNum={latestChapter}
        totalChapters={chapters.length}
        onShare={handleShare}
        onResume={handleResume}
        onUpload={() => setUploadOpen(true)}
      />

      {/* Optional resume bar shown below the hero on touch widths */}
      <div className="px-4 sm:px-6 mb-3 sm:hidden">
        <ContinueReadingBar
          resumeFrom={resumeFrom}
          onResume={handleResume}
        />
      </div>

      <WorkMembersPanel workId={workIdNum} />

      <LinkSuggestionPanel
        workId={workIdNum}
        ownMaterials={materials}
        targetLang={targetLang}
      />

      <WorkChapterList
        workId={workIdNum}
        chapters={chapters}
        targetLang={targetLang}
        loading={manifestsLoading}
        getSpawnState={spawnCtl.getSpawnState}
        onSpawn={handleSpawn}
        onAbort={handleAbort}
        onRetryTranslation={handleRetryTranslation}
        onOpenVersion={handleOpenVersion}
      />

      <UploadChapterDialog
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        workId={workIdNum}
        workTitle={workTitleForUpload}
        existing={existingChapterNumbers}
      />
    </div>
  )
}


export const Route = createFileRoute('/w/$workId')({
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
