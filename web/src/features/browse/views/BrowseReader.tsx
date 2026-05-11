import { useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from '@tanstack/react-router'
import { ArrowLeft, AlertTriangle } from 'lucide-react'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { fetchChapterPages, fetchMangaDetail } from '../manifest/runtime'
import { proxify } from '../proxy'
import { useLibrary } from '@features/library/store'
import type { InstalledSource, MangaDetail } from '../manifest/types'

// =============================================================================
// BrowseReader — raw chapter reader for /browse.
// Displays upstream page URLs directly (proxied for Referer/UA). No
// project required, no queue interaction. This is "Đọc thử" — instant
// access before deciding to import.
//
// The DA reader chrome (chrome='bare' route) provides scroll & safe
// areas; we render a continuous strip of <img> with proxy-aware src.
// =============================================================================

interface Props {
  source:     InstalledSource
  mangaUrl:   string
  chapterUrl: string
}

export function BrowseReader({ source, mangaUrl, chapterUrl }: Props) {
  const { manifest } = source

  const { data, isPending, isError, error } = useQuery({
    queryKey: ['browse', 'chapter', manifest.id, chapterUrl],
    queryFn:  () => fetchChapterPages(manifest, chapterUrl),
    // Chapter pages are immutable once published — long cache.
    staleTime: 15 * 60_000,
  })

  // Read the manga detail from React Query cache when possible. If
  // the user landed here from /browse/$source/manga/$mangaId, MangaPage
  // already populated this; multi-lang sources key by `language` so we
  // search all matching language entries and take the freshest.
  // Falls back to a fresh fetch (lang = first available) only when the
  // user deep-links to a chapter directly without visiting MangaPage.
  const qc = useQueryClient()
  const cached = qc.getQueriesData<MangaDetail | undefined>({
    queryKey: ['browse', 'manga', manifest.id, mangaUrl],
  })
  const cachedManga = cached
    .map(([, v]) => v)
    .find((v): v is MangaDetail => !!v) ?? null

  const { data: fetchedManga } = useQuery({
    queryKey: ['browse', 'manga', manifest.id, mangaUrl, ''],
    queryFn:  () => fetchMangaDetail(manifest, mangaUrl, { language: manifest.languages[0] ?? 'en' }),
    staleTime: 5 * 60_000,
    refetchOnMount: false,
    enabled: !cachedManga,
  })
  const manga = cachedManga ?? fetchedManga ?? null

  // Library bookkeeping — mark this chapter as the one the user is
  // currently reading. Triggers once per (mangaUrl, chapterUrl). The
  // store dedupes title/cover writes so this is cheap even when the
  // user hits the same chapter repeatedly (rage-clicks, refresh).
  const markChapterRead = useLibrary((s) => s.markChapterRead)
  useEffect(() => {
    if (!manga) return
    const ch = manga.chapters.find((c) => c.url === chapterUrl)
    markChapterRead({
      source:   manifest.id,
      mangaUrl: manga.url,
      title:    manga.title,
      cover:    manga.cover,
      chapter: {
        url:    chapterUrl,
        label:  ch?.label  ?? 'Đang đọc',
        number: ch?.number ?? '',
      },
    })
  }, [manga, chapterUrl, manifest.id, markChapterRead])

  return (
    <div className="min-h-dvh bg-bg">
      {/* slim toolbar — bare shell, no app sidebar */}
      <div
        className="sticky top-0 z-10 flex items-center gap-3 h-bar px-3 sm:px-5 bg-surface/95 backdrop-blur"
        style={{ paddingTop: 'var(--sait)' }}
      >
        <Link
          to="/browse/$source/manga/$mangaId"
          params={{ source: manifest.id, mangaId: encodeURIComponent(mangaUrl) }}
          className="inline-flex items-center gap-1.5 text-sm text-text-muted hover:text-text"
        >
          <ArrowLeft size={14} />
          Quay lại
        </Link>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-text-subtle truncate">{manifest.name} · Raw</p>
        </div>
        {data && (
          <span className="text-xs text-text-subtle tabular shrink-0">
            {data.pages.length} trang
          </span>
        )}
      </div>

      {isPending && (
        <div className="flex items-center justify-center py-32">
          <Spinner size={20} />
        </div>
      )}

      {isError && (
        <div className="px-4 py-16">
          <EmptyState
            icon={AlertTriangle}
            title="Không tải được chương"
            hint={(error as Error)?.message ?? 'Selector của nguồn có thể đã cũ.'}
          />
        </div>
      )}

      {data && data.pages.length === 0 && (
        <div className="px-4 py-16">
          <EmptyState
            icon={AlertTriangle}
            title="Không tìm thấy trang nào"
            hint="Selector .reader-img có thể không khớp với trang này."
          />
        </div>
      )}

      {data && data.pages.length > 0 && (
        <div className="max-w-3xl mx-auto">
          {data.pages.map((url, i) => (
            <img
              key={`${i}-${url}`}
              src={proxify(url)}
              alt={`Trang ${i + 1}`}
              loading="lazy"
              decoding="async"
              draggable={false}
              className="block w-full select-none"
            />
          ))}
          <div
            className="px-4 py-8 text-center text-xs text-text-subtle"
            style={{ paddingBottom: 'calc(2rem + var(--saib))' }}
          >
            Hết chương · {manifest.name}
          </div>
        </div>
      )}
    </div>
  )
}
