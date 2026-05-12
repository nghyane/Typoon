import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import {
  BookOpen, AlertTriangle, ArrowLeft, ExternalLink, Sparkles,
} from 'lucide-react'
import { useHeaderStore } from '../../../store/header'
import { useAutoTranslate, shouldTranslate } from '../autoTranslate'
import { useTranslated, useTranslatedBatch } from '../useTranslated'
import { Button } from '@shared/ui/Button'
import { Cover } from '@shared/ui/Cover'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { cn } from '@shared/lib/cn'
import { fetchMangaDetail } from '../manifest/runtime'
import { proxify } from '../proxy'
import { api, type ApiChapterTranslation, type ApiLibraryEntry } from '@shared/api/api'
import { useLibrary } from '@features/library/store'
import { useDefaultGuildId } from '@features/auth/useMe'
import { FollowButton } from '@features/library/views/LibraryCard'
import { SpawnDialog } from '@features/translate/SpawnDialog'
import { useTranslateProgress } from '@features/translate/useTranslateProgress'
import type { InstalledSource, MangaChapterRef, MangaDetail } from '../manifest/types'

// =============================================================================
// MangaPage — source manga detail surface.
//
// This is the per-source flavor of the future /title/{entry_id} hub.
// The user lands here when they tap a manga card from /browse; the
// page imports the material on first load and wires up everything
// the hub will later inherit: follow status, target-lang awareness,
// chapter overlay, inline spawn.
//
// Pro-design rules applied:
//   • One sticky hero, density 80px cover, action cluster right.
//   • Action verb is "Theo dõi" (FollowButton). Library status drives
//     the label — no `bookmarked` boolean leaks into UI.
//   • Chapter row is one widget: shows `[Đọc VN by @x]` when a
//     translation is available, `[Dịch]` when not. No filter pills.
//   • Lang picker is a single select beside the chapter count when
//     the manifest exposes multiple langs.
//   • Description collapses behind <details> so chapters ride near
//     the top — readers want list density, not blurb prose.
// =============================================================================

interface Props {
  source:   InstalledSource
  mangaUrl: string
}

export function MangaPage({ source, mangaUrl }: Props) {
  const { manifest } = source

  // Language picker — only meaningful when the source supports
  // multiple translation languages.
  const langs       = manifest.languages
  const isMultiLang = langs.length > 1
  const [language, setLanguage] = useState<string>(langs[0] ?? 'en')

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)

  const { data: manga, isPending, isError, error } = useQuery({
    queryKey:  ['browse', 'manga', manifest.id, mangaUrl, isMultiLang ? language : ''],
    queryFn:   () => fetchMangaDetail(manifest, mangaUrl, { language }),
    staleTime: 5 * 60_000,
  })

  // Narrow the picker to languages the page actually carries chapters
  // for, then auto-switch off an unavailable language.
  useEffect(() => {
    const avail = manga?.availableLanguages
    if (!avail || avail.length === 0) return
    if (!avail.includes(language) && avail[0]) setLanguage(avail[0])
  }, [manga?.availableLanguages, language])

  useEffect(() => {
    if (manga) {
      setHeader(manga.title, [{ label: manifest.name, to: '/browse/$source' }])
    } else {
      setHeader(manifest.name, [{ label: 'Duyệt nguồn', to: '/browse' }])
    }
    return () => clearHeader()
  }, [manga, manifest, setHeader, clearHeader])

  // Local reading history — keeps cross-source "Tiếp tục đọc" rail
  // accurate without a backend round-trip. Server library is the
  // source of truth for follow status; this is purely derived state.
  const recordView = useLibrary((s) => s.recordView)
  useEffect(() => {
    if (!manga) return
    const latest = manga.chapters[0]
    recordView({
      source:   manifest.id,
      mangaUrl: manga.url,
      title:    manga.title,
      cover:    manga.cover,
      latestChapter: latest
        ? { url: latest.url, label: latest.label, number: latest.number }
        : null,
    })
  }, [manga, manifest.id, recordView])

  if (isPending) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner size={20} />
      </div>
    )
  }
  if (isError || !manga) {
    return (
      <EmptyState
        icon={AlertTriangle}
        title="Không tải được truyện"
        hint={(error as Error)?.message ?? 'Trang có thể đã bị di chuyển.'}
      />
    )
  }

  return (
    <MangaContent
      source={source}
      manga={manga}
      language={language}
      langs={langs}
      isMultiLang={isMultiLang}
      setLanguage={setLanguage}
    />
  )
}

function MangaContent({
  source, manga, language, langs, isMultiLang, setLanguage,
}: {
  source:      InstalledSource
  manga:       MangaDetail
  language:    string
  langs:       string[]
  isMultiLang: boolean
  setLanguage: (l: string) => void
}) {
  const { manifest } = source
  const autoEnabled = useAutoTranslate((s) => s.enabled)
  const autoTarget  = useAutoTranslate((s) => s.target)

  // Eagerly import the material so the chapter rows know their
  // materialId for spawn. The same query is consumed by the library
  // lookup below — React Query dedupes on identical key.
  const { data: material } = useQuery({
    queryKey: ['material', 'import', manifest.id, manga.url],
    queryFn:  () => api.importMaterial({
      source:       manifest.id,
      upstream_ref: manga.url,
      title:        manga.title,
      cover_url:    manga.cover,
      description:  manga.description,
      author:       manga.author,
      status:       manga.status,
      languages:    manifest.languages,
      nsfw:         !!manifest.nsfw,
    }),
    staleTime: 5 * 60_000,
  })
  const materialId = material?.id ?? null

  // Active guild for translation visibility scope.
  const scopeGuildId = useDefaultGuildId()

  // Library entry (if any) for this material — drives FollowButton
  // state and `target_lang` override.
  const { data: entries = [] } = useQuery({
    queryKey:  ['library'],
    queryFn:   () => api.listLibrary(),
    staleTime: 30_000,
  })
  const entry = useMemo(() =>
    entries.find((e: ApiLibraryEntry) =>
      material != null
      && e.materials.some((m) => m.material_id === material.id)
    ),
    [entries, material],
  )

  // Reading language — entry's `target_lang` wins when set; otherwise
  // fall back to the global auto-translate target.
  const targetLang = entry?.target_lang ?? autoTarget

  // Translation overlay — keyed by manifest chapter URL.
  const upstreamUrls = useMemo(
    () => manga.chapters.map((c) => c.url),
    [manga.chapters],
  )
  const { data: overlay = {} } = useQuery({
    queryKey:  ['material', materialId, 'translation-overlay'],
    queryFn:   () => api.translationOverlay(materialId!, upstreamUrls),
    enabled:   materialId != null && upstreamUrls.length > 0,
    staleTime: 30_000,
  })

  // Title/description auto-translate is a separate concern from the
  // chapter pipeline — it lives in the local browser-translate hook.
  const useTr   = shouldTranslate(autoEnabled, targetLang, manifest.languages)
  const trTitle = useTranslated(manga.title,       targetLang, useTr)
  const trDesc  = useTranslated(manga.description, targetLang, useTr)

  const labels   = useMemo(() => manga.chapters.map((c) => c.label), [manga.chapters])
  const trLabels = useTranslatedBatch(labels, targetLang, useTr)

  const displayTitle = useTr && trTitle ? trTitle : manga.title
  const displayDesc  = useTr && trDesc  ? trDesc  : manga.description

  const firstChapter = manga.chapters[manga.chapters.length - 1] ?? manga.chapters[0]

  return (
    <div className="pb-16">
      {/* Mobile back */}
      <div className="sm:hidden px-4 pt-4">
        <Link
          to="/browse/$source"
          params={{ source: manifest.id }}
          className="inline-flex items-center gap-1.5 text-sm text-text-subtle hover:text-text"
        >
          <ArrowLeft size={14} />
          {manifest.name}
        </Link>
      </div>

      {/* Hero ────────────────────────────────────────────────────── */}
      <header className="px-4 sm:px-6 pt-4 sm:pt-6 pb-4 flex items-start gap-3 sm:gap-4">
        <Cover
          src={manga.cover ? proxify(manga.cover) : null}
          title={displayTitle}
          fontSize="text-xl"
          className="w-20 aspect-[2/3] rounded-md shrink-0"
        />
        <div className="flex-1 min-w-0">
          <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 sm:gap-3">
            <div className="min-w-0">
              <h1 className="text-lg sm:text-2xl font-semibold tracking-tight text-text line-clamp-2">
                {displayTitle}
              </h1>
              {useTr && trTitle && trTitle !== manga.title && (
                <p className="text-[11px] text-text-subtle mt-1 italic line-clamp-1">
                  {manga.title}
                </p>
              )}

              {/* Meta badge row */}
              <div className="flex items-center gap-2 mt-2 flex-wrap text-xs text-text-subtle">
                <span className="inline-flex items-center gap-1 h-[22px] px-2 rounded-xs bg-surface-2 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                  {manifest.languages[0]?.toUpperCase() ?? '?'}
                  <span className="text-text-subtle">→</span>
                  {targetLang.toUpperCase()}
                </span>
                {manga.author && <span>{manga.author}</span>}
                {manga.status && <span>· {manga.status}</span>}
                <a
                  href={manga.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 hover:text-text"
                >
                  {manifest.name}
                  <ExternalLink size={10} />
                </a>
                {manifest.nsfw && (
                  <span className="text-[11px] uppercase font-semibold px-1.5 py-0.5 rounded-xs bg-error/15 text-error-text">
                    NSFW
                  </span>
                )}
              </div>
            </div>

            {/* Action cluster — right side */}
            <div className="flex items-center gap-2 shrink-0 self-start">
              <FollowButton
                entryId={entry?.id}
                materialId={material?.id}
                title={manga.title}
                cover={manga.cover}
                targetLang={targetLang}
                status={entry?.status ?? null}
              />
              {firstChapter && (
                <Link
                  to="/browse/$source/manga/$mangaId/chapter/$chapterId"
                  params={{
                    source:    manifest.id,
                    mangaId:   encodeURIComponent(manga.url),
                    chapterId: encodeURIComponent(firstChapter.url),
                  }}
                >
                  <Button>
                    <BookOpen size={13} />
                    Đọc thử
                  </Button>
                </Link>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Description — collapsible */}
      {displayDesc && (
        <details className="px-4 sm:px-6 pb-4 group">
          <summary className="text-xs text-text-subtle cursor-pointer hover:text-text-muted list-none flex items-center gap-1.5">
            <span>Mô tả</span>
            <span className="group-open:rotate-180 transition-transform">▾</span>
          </summary>
          <p className="text-sm text-text-muted leading-relaxed whitespace-pre-line mt-2 max-w-2xl">
            {displayDesc}
          </p>
        </details>
      )}

      {/* Chapter section header */}
      <div className="px-4 sm:px-6 flex items-center justify-between gap-3 mt-2 mb-2">
        <h2 className="text-sm font-semibold text-text">
          {manga.chapters.length} chương
        </h2>
        {isMultiLang && (
          <LangPicker
            value={language}
            options={manga.availableLanguages ?? langs}
            onChange={setLanguage}
          />
        )}
      </div>

      {/* Chapter list */}
      <div className="px-4 sm:px-6">
        {manga.chapters.length === 0 ? (
          <EmptyState
            icon={BookOpen}
            title="Chưa có chương đọc được"
            hint={isMultiLang
              ? `Không có chương ${language.toUpperCase()} đọc được tại nguồn. Hãy thử ngôn ngữ khác.`
              : 'Nguồn có thể chưa cập nhật hoặc đã đổi cấu trúc.'}
          />
        ) : (
          <ul className="rounded-md bg-surface divide-y divide-border-soft overflow-hidden">
            {manga.chapters.map((c, i) => (
              <ChapterRow
                key={c.id}
                manifestId={manifest.id}
                mangaId={manga.url}
                materialId={materialId}
                chapter={c}
                translatedLabel={useTr ? trLabels[i] : null}
                targetLang={targetLang}
                scopeGuildId={scopeGuildId}
                manga={manga}
                overlay={overlay[c.url] ?? []}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}


// ── Lang picker ─────────────────────────────────────────────────────

function LangPicker({
  value, options, onChange,
}: {
  value:    string
  options:  string[]
  onChange: (v: string) => void
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-7 px-2 rounded-sm bg-surface-2 text-xs text-text border-0 cursor-pointer"
    >
      {options.map((l) => (
        <option key={l} value={l}>{l.toUpperCase()}</option>
      ))}
    </select>
  )
}


// ── Chapter row ─────────────────────────────────────────────────────
//
// One verb at a time. When a translation exists at the user's
// target_lang (or any done variant), the row shows `[Đọc VN by @x]`
// and a small `Sparkles` button to fork a personal copy. When nothing
// is available, the row collapses to `[Raw] [Dịch]`. While a spawn
// is running, the chip replaces both with a live progress badge.

function ChapterRow({
  manifestId, mangaId, materialId, chapter, translatedLabel, targetLang,
  scopeGuildId, manga, overlay,
}: {
  manifestId:       string
  mangaId:          string
  /** Server material row id (resolved upstream via importMaterial).
   *  When null, the row only shows "Đọc raw" — spawn is disabled until
   *  the import settles. */
  materialId:       number | null
  chapter:          MangaChapterRef
  translatedLabel?: string | null
  targetLang:       string
  scopeGuildId:     string | null
  manga:            MangaDetail
  /** Translations the viewer can read on this chapter, keyed off
   *  manifest upstream_url. */
  overlay:          ApiChapterTranslation[]
}) {
  const showTr = translatedLabel && translatedLabel !== chapter.label
  const [spawnOpen, setSpawnOpen] = useState(false)
  const [pendingId, setPendingId] = useState<number | null>(null)
  const progress = useTranslateProgress(pendingId, pendingId !== null)
  const qc = useQueryClient()

  if (progress?.state === 'done' && pendingId !== null) {
    qc.invalidateQueries({ queryKey: ['material'] })
  }

  const readable = useMemo(
    () => pickReadable(overlay, targetLang),
    [overlay, targetLang],
  )

  const spawn = useMutation({
    mutationFn: () =>
      api.spawnTranslate({
        chapter_ref: {
          material_id:  materialId!,
          upstream_url: chapter.url,
          number:       chapter.number,
          label:        chapter.label,
        },
        target_lang:    targetLang,
        visibility:     scopeGuildId ? 'guild' : 'private',
        scope_guild_id: scopeGuildId,
      }),
    onSuccess: (result) => {
      setPendingId(result.translation_id)
      if (result.cache_hit) {
        qc.invalidateQueries({ queryKey: ['material'] })
      }
    },
  })

  return (
    <li className="flex items-center gap-3 px-3 py-2.5 hover:bg-hover transition-colors">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5">
          <p className="text-sm text-text truncate">
            {showTr ? translatedLabel : chapter.label}
          </p>
          {overlay.length > 0 && <OverlayBadges overlay={overlay} />}
        </div>
        {showTr && (
          <p className="text-text-subtle font-normal text-[11px] italic mt-0.5 truncate">
            {chapter.label}
          </p>
        )}
        <div className="flex items-center gap-2 text-[11px] text-text-subtle mt-0.5">
          {chapter.language && (
            <span className="uppercase font-semibold bg-surface-2 px-1 py-0.5 rounded-xs">
              {chapter.language}
            </span>
          )}
          {chapter.date && <span className="tabular">{chapter.date}</span>}
          {readable?.creator_name && (
            <span className="truncate" title={`Bản của ${readable.creator_name}`}>
              · @{readable.creator_name}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1.5 shrink-0">
        {pendingId !== null && progress ? (
          <ProgressChip progress={progress} />
        ) : readable ? (
          <>
            <Link
              to="/browse/$source/manga/$mangaId/chapter/$chapterId"
              params={{
                source:    manifestId,
                mangaId:   encodeURIComponent(mangaId),
                chapterId: encodeURIComponent(chapter.url),
              }}
              search={{ tx: readable.id } as never}
              className="inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] font-medium bg-success/15 text-success-text hover:bg-success/25 cursor-pointer"
              title={readable.creator_name ? `Bản dịch của ${readable.creator_name}` : 'Bản dịch sẵn'}
            >
              <BookOpen size={11} />
              Đọc {readable.target_lang.toUpperCase()}
            </Link>
            <button
              type="button"
              onClick={() => { if (materialId) setSpawnOpen(true) }}
              disabled={!materialId}
              className="inline-flex items-center justify-center size-7 rounded-sm text-text-muted hover:bg-surface-2 hover:text-text cursor-pointer disabled:opacity-50"
              title="Tạo bản dịch riêng"
            >
              <Sparkles size={11} />
            </button>
          </>
        ) : (
          <>
            <Link
              to="/browse/$source/manga/$mangaId/chapter/$chapterId"
              params={{
                source:    manifestId,
                mangaId:   encodeURIComponent(mangaId),
                chapterId: encodeURIComponent(chapter.url),
              }}
              className="inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] text-text-muted hover:bg-surface-2 hover:text-text"
            >
              <BookOpen size={11} />
              Raw
            </Link>
            <button
              type="button"
              onClick={() => {
                if (!materialId) return
                if (scopeGuildId) spawn.mutate()
                else setSpawnOpen(true)
              }}
              disabled={!materialId || spawn.isPending}
              className={cn(
                'inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] font-medium cursor-pointer',
                'bg-accent text-accent-fg hover:bg-accent-strong',
                (!materialId || spawn.isPending) && 'opacity-60 cursor-wait',
              )}
              title="Dịch chương này"
            >
              <Sparkles size={11} />
              Dịch
            </button>
          </>
        )}
      </div>

      <SpawnDialog
        open={spawnOpen}
        onClose={() => setSpawnOpen(false)}
        chapterId={0}
        title={`${manga.title} · Ch ${chapter.number} → ${targetLang.toUpperCase()}`}
        targetLang={targetLang}
        scopeGuildId={scopeGuildId}
        onSpawned={(r) => setPendingId(r.translation_id)}
      />
    </li>
  )
}


function pickReadable(
  overlay:    ApiChapterTranslation[],
  targetLang: string,
): ApiChapterTranslation | null {
  if (overlay.length === 0) return null
  const done = overlay.filter((t) => t.state === 'done')
  return (
    done.find((t) => t.target_lang === targetLang)
    ?? done[0]
    ?? null
  )
}


function OverlayBadges({
  overlay,
}: {
  overlay: ApiChapterTranslation[]
}) {
  // Distinct done languages — collapse duplicates so 3 VN translations
  // show "[VN]" once not three times.
  const langs = new Map<string, ApiChapterTranslation>()
  for (const t of overlay) {
    if (t.state !== 'done') continue
    if (!langs.has(t.target_lang)) langs.set(t.target_lang, t)
  }
  if (langs.size === 0) {
    const running = overlay.find((t) => t.state === 'running')
    if (!running) return null
    return (
      <span className="text-[11px] uppercase font-semibold px-1 py-0.5 rounded-xs bg-accent/15 text-accent-text">
        đang dịch
      </span>
    )
  }
  const shown = [...langs.values()].slice(0, 3)
  return (
    <div className="flex items-center gap-0.5">
      {shown.map((t) => (
        <span
          key={t.id}
          className="text-[11px] uppercase font-semibold px-1 py-0.5 rounded-xs bg-success/15 text-success-text"
          title={t.creator_name ? `Bản của ${t.creator_name}` : 'Bản dịch sẵn'}
        >
          {t.target_lang}
        </span>
      ))}
      {langs.size > shown.length && (
        <span className="text-[11px] text-text-subtle">+{langs.size - shown.length}</span>
      )}
    </div>
  )
}


function ProgressChip({
  progress,
}: {
  progress: ReturnType<typeof useTranslateProgress>
}) {
  if (!progress) return null
  const label = (() => {
    if (progress.state === 'error') return `Lỗi: ${progress.error?.slice(0, 40) ?? '?'}`
    if (progress.state === 'done')  return 'Xong'
    if (progress.total === 0)       return `${progress.stage || 'pending'}…`
    const pct = Math.round((progress.index / progress.total) * 100)
    return `${progress.stage} ${pct}%`
  })()
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 h-7 px-2.5 rounded-sm text-[11px] font-medium',
        progress.state === 'error'
          ? 'bg-error/15 text-error-text'
          : progress.state === 'done'
          ? 'bg-success/15 text-success-text'
          : 'bg-accent/15 text-accent-text',
      )}
    >
      <Sparkles size={11} className={progress.state === 'done' ? '' : 'animate-pulse'} />
      {label}
    </span>
  )
}
