import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { BookOpen, AlertTriangle, ArrowLeft, ExternalLink, Languages, Bookmark } from 'lucide-react'
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
import { api } from '@shared/api/api'
import { useLibrary } from '@features/library/store'
import type { InstalledSource, MangaChapterRef, MangaDetail } from '../manifest/types'

interface Props {
  source:   InstalledSource
  mangaUrl: string
}

export function MangaPage({ source, mangaUrl }: Props) {
  const { manifest } = source

  // Language picker — only meaningful when the source supports
  // multiple translation languages.
  const langs = manifest.languages
  const isMultiLang = langs.length > 1
  const [language, setLanguage] = useState<string>(langs[0] ?? 'en')

  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)

  const { data: manga, isPending, isError, error } = useQuery({
    queryKey: ['browse', 'manga', manifest.id, mangaUrl, isMultiLang ? language : ''],
    queryFn:  () => fetchMangaDetail(manifest, mangaUrl, { language }),
    staleTime: 5 * 60_000,
  })

  // When the manga page returns the set of languages it actually has
  // chapters for, narrow the picker to that intersection. If the
  // user's current pick isn't in the list, auto-switch to the first
  // available so they see chapters immediately instead of an empty
  // state.
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

  // Library — record view (title/cover/latestChapter snapshot) so:
  //   1. This manga shows up in the cross-source "Đang đọc" list
  //   2. New-chapter detection can compare latest vs lastChapterRead
  // Browser-side via DA proxy = free; no backend involvement.
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
  const setAuto     = useAutoTranslate((s) => s.setEnabled)

  // Translate only when user wants it AND the source is not already
  // in the user's target language.
  const useTr = shouldTranslate(autoEnabled, autoTarget, manifest.languages)

  const trTitle = useTranslated(manga.title, autoTarget, useTr)
  const trDesc  = useTranslated(manga.description, autoTarget, useTr)

  // Chapter labels — batch one shot. Keep stable order so cache hits.
  const labels = useMemo(() => manga.chapters.map((c) => c.label), [manga.chapters])
  const trLabels = useTranslatedBatch(labels, autoTarget, useTr)

  const displayTitle = useTr && trTitle ? trTitle : manga.title
  const displayDesc  = useTr && trDesc  ? trDesc  : manga.description

  const firstChapter = manga.chapters[manga.chapters.length - 1] ?? manga.chapters[0]

  return (
    <div className="pb-16">
      {/* mobile back to source */}
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

      {/* hero */}
      <div className="px-4 sm:px-6 pt-4 sm:pt-6 pb-5 flex items-start gap-4 sm:gap-6">
        <Cover
          src={manga.cover ? proxify(manga.cover) : null}
          title={displayTitle}
          className="w-24 sm:w-32 aspect-[2/3] rounded-md shrink-0"
        />
        <div className="flex-1 min-w-0 pt-1">
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-text leading-tight">
            {displayTitle}
          </h1>
          {useTr && trTitle && trTitle !== manga.title && (
            <p className="text-[11px] text-text-subtle mt-1 italic">
              {manga.title}
            </p>
          )}
          <p className="text-xs text-text-subtle mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
            {manga.author && <span>{manga.author}</span>}
            {manga.author && manga.status && <span>·</span>}
            {manga.status && <span>{manga.status}</span>}
            {(manga.author || manga.status) && <span>·</span>}
            <a
              href={manga.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 hover:text-text"
            >
              {manifest.name}
              <ExternalLink size={10} />
            </a>
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            <SaveButton source={source} manga={manga} />
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
            {!manifest.languages.includes(autoTarget) && (
              <Button
                size="sm"
                onClick={() => setAuto(!autoEnabled)}
                title={`${autoEnabled ? 'Tắt' : 'Bật'} tự dịch sang ${autoTarget.toUpperCase()}`}
              >
                <Languages size={12} />
                {autoEnabled ? `Đã dịch ${autoTarget.toUpperCase()}` : 'Tự dịch'}
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* description */}
      {displayDesc && (
        <div className="px-4 sm:px-6 pb-5">
          <p className="text-sm text-text-muted leading-relaxed whitespace-pre-line">
            {displayDesc}
          </p>
        </div>
      )}

      {/* chapter list */}
      <div className="px-4 sm:px-6">
        <div className="flex items-center justify-between mb-3 gap-3">
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

        {manga.chapters.length === 0 ? (
          <EmptyState
            icon={BookOpen}
            title="Chưa có chương đọc được"
            hint={isMultiLang
              ? `Không có chương ${language.toUpperCase()} đọc được tại nguồn. Có thể bản dịch được host ở site khác (đã mua bản quyền). Hãy thử ngôn ngữ khác.`
              : 'Nguồn có thể chưa cập nhật hoặc đã đổi cấu trúc.'}
          />
        ) : (
          <ul className="rounded-md bg-surface divide-y divide-border-soft overflow-hidden">
            {manga.chapters.map((c, i) => (
              <ChapterRow
                key={c.id}
                manifestId={manifest.id}
                mangaId={manga.url}
                chapter={c}
                translatedLabel={useTr ? trLabels[i] : null}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

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

function ChapterRow({
  manifestId, mangaId, chapter, translatedLabel,
}: {
  manifestId:       string
  mangaId:          string
  chapter:          MangaChapterRef
  translatedLabel?: string | null
}) {
  const showTr = translatedLabel && translatedLabel !== chapter.label
  return (
    <li>
      <Link
        to="/browse/$source/manga/$mangaId/chapter/$chapterId"
        params={{
          source:    manifestId,
          mangaId:   encodeURIComponent(mangaId),
          chapterId: encodeURIComponent(chapter.url),
        }}
        className={cn(
          'flex items-center gap-3 px-3 py-2.5 hover:bg-hover transition-colors cursor-pointer',
        )}
      >
        <span className="flex-1 min-w-0 text-sm text-text truncate">
          {showTr ? translatedLabel : chapter.label}
          {showTr && (
            <span className="text-text-subtle font-normal ml-2 text-[11px] italic">
              {chapter.label}
            </span>
          )}
        </span>
        {chapter.language && (
          <span className="text-[10px] uppercase font-semibold text-text-subtle bg-surface-2 px-1.5 py-0.5 rounded-xs shrink-0">
            {chapter.language}
          </span>
        )}
        {chapter.date && (
          <span className="text-xs text-text-subtle tabular shrink-0">{chapter.date}</span>
        )}
      </Link>
    </li>
  )
}


// =============================================================================
// SaveButton — "Lưu vào thư viện" for the manga detail hero.
//
// Source-backed manga need a material row before we can attach a
// library_entry. The first click does an idempotent
// /api/material/import with whatever the manifest gave us; subsequent
// clicks toggle the bookmark flag on the existing entry.
//
// Mirroring state from /api/library means N entries on this manga
// (same user) collapse to one — the library list is the source of
// truth, not local guesses.
// =============================================================================

function SaveButton({
  source, manga,
}: {
  source: InstalledSource
  manga:  MangaDetail
}) {
  const qc = useQueryClient()
  const { manifest } = source

  // We pull the user's library to see whether this material already
  // has an entry — needed for the bookmark icon state and to know
  // which entry id to PATCH on toggle.
  const { data: entries = [] } = useQuery({
    queryKey: ['library'],
    queryFn:  () => api.listLibrary(),
    staleTime: 30_000,
  })

  // Look up the material id once we've imported. Cached so subsequent
  // toggles don't re-import on every render.
  const { data: material } = useQuery({
    queryKey: ['material', manifest.id, manga.url],
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

  const entry = useMemo(
    () => entries.find((e) =>
      material != null
      && e.materials.some((m) => m.material_id === material.id)
    ),
    [entries, material],
  )

  const mutation = useMutation({
    mutationFn: async () => {
      if (!material) return
      if (entry) {
        await api.patchLibraryEntry(entry.id, {
          bookmarked: !entry.bookmarked,
        })
      } else {
        const created = await api.createLibraryEntry({
          material_id: material.id,
          title:       manga.title,
          cover_url:   manga.cover,
        })
        if (!created.bookmarked) {
          await api.patchLibraryEntry(created.id, { bookmarked: true })
        }
      }
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
    },
  })

  const on = !!entry?.bookmarked
  const disabled = mutation.isPending || material == null

  return (
    <button
      type="button"
      onClick={() => mutation.mutate()}
      disabled={disabled}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm cursor-pointer transition-colors h-8 px-3 text-[13px]',
        on
          ? 'bg-warning/15 text-warning-text hover:bg-warning/25'
          : 'bg-surface-2 text-text-muted hover:bg-hover hover:text-text',
        disabled && 'opacity-60 cursor-wait',
      )}
      title={on ? 'Bỏ lưu khỏi thư viện' : 'Lưu vào thư viện'}
    >
      <Bookmark
        size={13}
        className={on ? 'fill-warning text-warning' : ''}
      />
      {on ? 'Đã lưu' : 'Lưu'}
    </button>
  )
}
