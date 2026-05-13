// /m/$materialId — canonical manga page.
//
// Single surface across feed clicks, library clicks, share links, and
// search results. Schema 19 made manga a community resource — there
// is no per-user title hub anymore.
//
// State machine for the hero action:
//   • viewer has no library entry → "+ Theo dõi" (primary, accent)
//   • viewer has an entry         → status dropdown (status verb + ▾)
//
// Bookmark is explicit. Opening this page or reading a chapter does
// NOT create an entry. Reading history is a separate concept tracked
// via /api/me/reading; the home page surfaces "Tiếp tục đọc" from it.

import { useEffect, useMemo, useState } from 'react'
import { createFileRoute, redirect } from '@tanstack/react-router'
import { useQuery, keepPreviousData } from '@tanstack/react-query'
import {
  AlertTriangle, BookOpen, ChevronDown, Clock,
  Languages, MoreHorizontal, Share2, Sparkles,
} from 'lucide-react'

import { api } from '@shared/api/api'
import type { ApiMaterial } from '@shared/api/api'
import { Button } from '@shared/ui/Button'
import { Cover, coverUrl } from '@shared/ui/Cover'
import { EmptyState } from '@shared/ui/EmptyState'
import { Spinner } from '@shared/ui/primitives'
import { timeAgo } from '@shared/lib/time'
import { cn } from '@shared/lib/cn'
import { useSources } from '@features/browse/sources'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { BookmarkButton } from '@features/manga/BookmarkButton'
import {
  ChapterListView, SpawnRow, useChapterSelection,
} from '@features/title/chapter-list'
import { useLocalChapterListState } from '@features/title/chapterListState'
import {
  chapterStatus, mergeChapters, type HubChapter,
} from '@features/title/mergeChapters'
import { useHeaderStore } from '../store/header'


function MangaPage() {
  const { materialId } = Route.useParams()
  const id = Number(materialId)

  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const detailQ = useQuery({
    queryKey: ['material', 'detail', id],
    queryFn:  () => api.getMaterial(id),
    enabled:  Number.isFinite(id) && id > 0,
    staleTime: 30_000,
    placeholderData: keepPreviousData,
    refetchInterval: (q: { state: { data?: { chapters?: { translations?: { state?: string }[] }[] } } }) => {
      const chs = q.state.data?.chapters ?? []
      const inFlight = chs.some((c) =>
        (c.translations ?? []).some((t) => t.state === 'pending' || t.state === 'running'),
      )
      return inFlight ? 5_000 : false
    },
  })

  const installed = useSources((s) => s.sources)
  const material  = detailQ.data?.material ?? null
  const source    = material?.source ? (installed[material.source] ?? null) : null
  const manifestQ = useQuery({
    queryKey: ['manifest', 'detail', source?.manifest.id, material?.upstream_ref],
    queryFn:  () => fetchMangaDetail(source!.manifest, material!.upstream_ref!),
    enabled:  source !== null && !!material?.upstream_ref,
    staleTime: 5 * 60_000,
    retry:     false,
    placeholderData: keepPreviousData,
  })

  // Manifest chapters live in plugin runtime only. The server learns
  // about a chapter on-demand (spawn/upload/translate) — see Commit 1b
  // for work_chapter materialise; for now reading raw chapters does
  // not record history.

  const chapters: HubChapter[] = useMemo(() => {
    if (!detailQ.data) return []
    return mergeChapters([
      { detail: detailQ.data, source, manifest: manifestQ.data ?? null },
    ])
  }, [detailQ.data, source, manifestQ.data])

  const latestChapterNum = useMemo(() => {
    if (chapters.length === 0) return null
    return chapters[0].number
  }, [chapters])

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  if (detailQ.isPending) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (detailQ.error || !detailQ.data) {
    return (
      <div className="px-4 sm:px-6 pt-12">
        <EmptyState
          icon={AlertTriangle}
          title="Không tải được"
          hint={(detailQ.error as Error)?.message ?? 'Manga không tồn tại hoặc đã bị xoá.'}
        />
      </div>
    )
  }

  return (
    <div className="pb-16">
      <Hero
        material={detailQ.data.material}
        viewerEntry={detailQ.data.viewer_entry}
        latestChapterNum={latestChapterNum}
      />
      <ChapterListBody
        chapters={chapters}
        material={detailQ.data.material}
        loading={manifestQ.isPending && manifestQ.fetchStatus !== 'idle'}
      />
    </div>
  )
}


// ── Hero ────────────────────────────────────────────────────────────


function Hero({
  material, viewerEntry, latestChapterNum,
}: {
  material:        ApiMaterial
  viewerEntry:     { entry_id: number; status: string } | null
  latestChapterNum: string | null
}) {
  const [descExpanded, setDescExpanded] = useState(false)
  const coverSrc = coverUrl(material.cover_url, material.updated_at)
  const descLong = (material.description?.length ?? 0) > 200

  const handleShare = async () => {
    const url = `${window.location.origin}/m/${material.id}`
    if (navigator.share) {
      try { await navigator.share({ title: material.title, url }) } catch { /* share dialog dismissed */ }
    } else {
      void navigator.clipboard?.writeText(url)
    }
  }

  return (
    <div className="relative overflow-hidden">
      {/* Blurred cover background */}
      {coverSrc && (
        <div className="absolute inset-0 -top-4 -bottom-4 overflow-hidden pointer-events-none select-none">
          <img
            src={coverSrc}
            alt=""
            className="w-full h-full object-cover blur-[60px] scale-110 opacity-40"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-bg/60 via-bg/30 to-bg" />
        </div>
      )}

      {/* Hero content */}
      <div className="relative px-4 sm:px-6 pt-5 sm:pt-7 pb-5 sm:pb-6 flex items-start gap-4 sm:gap-6">
        {/* Cover — 2:3 aspect ratio, larger on desktop */}
        <Cover
          src={coverSrc}
          title={material.title}
          fontSize="text-2xl sm:text-3xl"
          className="w-[96px] sm:w-[140px] aspect-[2/3] rounded-md shrink-0 shadow-lg shadow-black/30"
        />

        <div className="flex-1 min-w-0 pt-0.5">
          {/* Title */}
          <h1 className="text-xl sm:text-3xl font-bold tracking-tight text-text leading-tight">
            {material.title}
          </h1>

          {/* Native + alt titles */}
          {(material.title_native || (material.title_alt?.length ?? 0) > 0) && (
            <div className="mt-1.5 flex items-center gap-2 flex-wrap">
              {material.title_native && (
                <span className="text-sm text-text-muted inline-flex items-center gap-1">
                  <Languages size={12} className="text-text-subtle shrink-0" />
                  {material.title_native}
                </span>
              )}
              {material.title_alt?.slice(0, 2).map((alt, i) => (
                <span key={i} className="text-sm text-text-subtle">
                  {material.title_native || i > 0 ? '· ' : ''}
                  {alt}
                </span>
              ))}
              {(material.title_alt?.length ?? 0) > 2 && (
                <span className="text-xs text-text-subtle">
                  +{material.title_alt!.length - 2}
                </span>
              )}
            </div>
          )}

          {/* Author / source */}
          <div className="mt-1 flex items-center gap-1.5 text-xs text-text-subtle">
            {material.languages[0] && (
              <span className="inline-flex items-center h-[20px] px-1.5 rounded-xs bg-surface-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                {material.languages[0].toUpperCase()}
              </span>
            )}
            {material.author && (
              <>
                <span className="text-text-muted">{material.author}</span>
                <span className="opacity-40">·</span>
              </>
            )}
            {material.source && (
              <span className="text-text-subtle">{material.source}</span>
            )}
          </div>

          {/* Action buttons */}
          <div className="mt-4 flex items-center gap-2 flex-wrap">
            {/* Đọc tiếp — scroll to chapter list */}
            {latestChapterNum && (
              <Button
                size="md"
                variant="primary"
                className="gap-1.5 font-semibold"
                onClick={() => {
                  document.getElementById('chapter-list')?.scrollIntoView({ behavior: 'smooth' })
                }}
              >
                <BookOpen size={14} />
                Đọc tiếp ch. {latestChapterNum}
              </Button>
            )}

            <BookmarkButton
              materialId={material.id}
              title={material.title}
              cover={material.cover_url}
              entryId={viewerEntry?.entry_id ?? null}
              status={
                (viewerEntry?.status ?? null) as
                  | 'reading' | 'plan' | 'done' | 'dropped' | null
              }
            />

            <Button
              size="md"
              variant="secondary"
              onClick={handleShare}
              className="gap-1.5"
            >
              <Share2 size={13} />
              Chia sẻ
            </Button>

            <Button size="md" variant="secondary" icon>
              <MoreHorizontal size={15} />
            </Button>
          </div>

          {/* Meta chips */}
          <div className="mt-3 flex items-center gap-3 flex-wrap text-xs text-text-muted">
            {material.nsfw && (
              <span className="inline-flex items-center gap-1 text-error-text font-semibold">
                <AlertTriangle size={11} /> NSFW
              </span>
            )}
            {material.status && (
              <span className="inline-flex items-center gap-1">
                <BookOpen size={11} />
                {material.status}
              </span>
            )}
            {material.updated_at && (
              <span className="inline-flex items-center gap-1">
                <Clock size={11} />
                Cập nhật {timeAgo(material.updated_at)}
              </span>
            )}
          </div>

          {/* Description with expand/collapse */}
          {material.description && (
            <div className="mt-3">
              <p
                className={cn(
                  'text-sm text-text-muted leading-relaxed max-w-2xl',
                  !descExpanded && 'line-clamp-4',
                )}
              >
                {material.description}
              </p>
              {descLong && !descExpanded && (
                <button
                  type="button"
                  onClick={() => setDescExpanded(true)}
                  className="inline-flex items-center gap-1 mt-1 text-xs text-accent-text hover:text-accent-strong transition-colors cursor-pointer"
                >
                  Đọc thêm <ChevronDown size={12} />
                </button>
              )}
              {descExpanded && (
                <button
                  type="button"
                  onClick={() => setDescExpanded(false)}
                  className="inline-flex items-center gap-1 mt-1 text-xs text-text-subtle hover:text-text-muted transition-colors cursor-pointer"
                >
                  Thu gọn
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


// ── Chapter list body ───────────────────────────────────────────────


function ChapterListBody({
  chapters, material, loading,
}: {
  chapters:  HubChapter[]
  material:  ApiMaterial
  loading:   boolean
}) {
  const state     = useLocalChapterListState()
  const selection = useChapterSelection(chapters)

  // Target lang for spawn defaults to the viewer-preferred one.
  // No per-user picker on this surface yet — VI by default for the
  // beta community. Future slice: read from user profile / per-manga
  // override.
  const targetLang = 'vi'

  const eligibleBulk = useMemo(() => {
    let n = 0
    for (const c of chapters) {
      if (!selection.has(c.number)) continue
      if (chapterStatus(c, targetLang) === 'raw') n++
    }
    return n
  }, [chapters, selection, targetLang])

  return (
    <ChapterListView
      chapters={chapters}
      targetLang={targetLang}
      loading={loading}
      state={state}
      materials={[]}
      selection={selection}
      eligibleBulk={eligibleBulk}
      bulkAction={
        <Button
          variant="primary" size="sm" disabled
          title="Bulk spawn sẽ wire ở slice tiếp"
        >
          <Sparkles size={12} />
          Dịch ({eligibleBulk})
        </Button>
      }
      renderRow={(c) => (
        <SpawnRow
          key={c.number}
          chapter={c}
          targetLang={targetLang}
          materialTitle={material.title}
          selection={selection}
        />
      )}
    />
  )
}


// ── Route ───────────────────────────────────────────────────────────


export const Route = createFileRoute('/m/$materialId')({
  beforeLoad: ({ params }) => {
    const id = Number(params.materialId)
    if (!Number.isFinite(id) || id <= 0) {
      throw redirect({ to: '/' })
    }
  },
  component: MangaPage,
})
