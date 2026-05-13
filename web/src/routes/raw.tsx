// Raw chapter reader — /raw?source=<id>&url=<encoded chapter URL>
//
// Hits the manifest at runtime to resolve page URLs, then renders a
// continuous strip of <img src={proxify(url)}> so the user reads
// without leaving the app. Mirrors the translated reader's chrome
// (bare shell, sticky toolbar) so switching between Đọc VI and Đọc raw
// feels consistent.
//
// Why a route + search params instead of /raw/$source/$url:
// chapter URLs are arbitrary upstream HTTPS strings; encoding them as
// a path segment trips both the router parser and Cloudflare's edge
// path normalization. Search params keep the URL flat and the value
// readable in dev tools.

import { createFileRoute } from '@tanstack/react-router'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, ChevronLeft } from 'lucide-react'
import { api } from '@shared/api/api'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { cn } from '@shared/lib/cn'
import { fetchChapterPages } from '@features/browse/manifest/runtime'
import { proxify } from '@features/browse/proxy'
import { useSources } from '@features/browse/sources'
import {
  clearReadingPresence, lockPortrait, setReadingPresence, unlockOrientation,
} from '@shared/discord/presence'

interface SearchParams {
  source: string
  url:    string
  /** Optional display context — passed in so the toolbar doesn't have
   *  to re-fetch the manga detail just to render a breadcrumb. */
  title?:      string
  number?:     string
  label?:      string
  /** When the raw is opened from a MangaPage, the SPA already knows
   *  which material drove the navigation. Used to record reading
   *  history at Work-chapter scope (server materialises the
   *  work_chapter on demand). Absent → no history recorded. */
  materialId?: number
  /** Manifest-normalised key for the chapter. Required alongside
   *  `materialId` to record history; absent → skip recording. */
  numberNorm?: string
}

function RawReaderPage() {
  const {
    source, url, title, number, label, materialId, numberNorm,
  } = Route.useSearch()
  const sources = useSources((s) => s.sources)
  const installed = sources[source] ?? null

  const { data, isPending, isError, error } = useQuery({
    queryKey: ['raw', 'chapter', source, url],
    queryFn:  () => fetchChapterPages(installed!.manifest, url),
    // Chapter pages are immutable once published — long cache.
    staleTime: 15 * 60_000,
    enabled:   installed !== null,
  })

  // Reading history — fire-and-forget when the raw chapter resolves.
  // Server materialises the Work chapter on demand from (material,
  // number_norm) so history dedupes across sources of the same Work.
  // No-op when materialId / numberNorm absent (raw opened from a
  // direct URL outside the MangaPage flow).
  const recordedRef = useRef<string | null>(null)
  const recordReading = useMutation({
    mutationFn: api.recordRawReading,
  })
  useEffect(() => {
    if (materialId == null || !number || !numberNorm) return
    const key = `${materialId}:${numberNorm}`
    if (recordedRef.current === key) return
    recordedRef.current = key
    recordReading.mutate({
      material_id: materialId,
      number,
      number_norm: numberNorm,
      label:       label ?? null,
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps -- recordReading is stable
  }, [materialId, number, numberNorm, label])

  // Discord Activity — same lifecycle as translated reader.
  useEffect(() => {
    if (!title || !number) return
    setReadingPresence({
      projectTitle:  title,
      chapterNumber: number,
      chapterTitle:  null,
    })
  }, [title, number])
  useEffect(() => {
    lockPortrait()
    return () => {
      unlockOrientation()
      clearReadingPresence()
    }
  }, [])

  if (!installed) {
    return (
      <div className="px-4 py-16">
        <EmptyState
          icon={AlertTriangle}
          title={`Nguồn "${source}" chưa được cài`}
          hint="Vào Cài đặt → Nguồn để cài hoặc cập nhật."
        />
      </div>
    )
  }

  return (
    <div className="min-h-dvh bg-bg">
      <RawToolbar
        sourceName={installed.manifest.name}
        title={title ?? ''}
        number={number ?? ''}
        total={data?.pages.length ?? 0}
        onBack={() => window.history.back()}
      />

      {isPending && (
        <div className="flex items-center justify-center py-24">
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
            hint="Manifest trả về danh sách rỗng."
          />
        </div>
      )}

      {data && data.pages.length > 0 && (
        <div className="max-w-3xl mx-auto">
          {data.pages.map((pageUrl, i) => (
            <img
              key={`${i}-${pageUrl}`}
              src={proxify(pageUrl)}
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
            Hết chương · {installed.manifest.name}
          </div>
        </div>
      )}
    </div>
  )
}


// ── Toolbar ──────────────────────────────────────────────────────────

function RawToolbar({
  sourceName, title, number, total, onBack,
}: {
  sourceName: string
  title:      string
  number:     string
  total:      number
  onBack:     () => void
}) {
  const [hidden, setHidden] = useState(false)
  useEffect(() => {
    let lastY = window.scrollY
    let raf: number | null = null
    const onScroll = () => {
      if (raf !== null) return
      raf = requestAnimationFrame(() => {
        const y = window.scrollY
        const dy = y - lastY
        if (Math.abs(dy) > 8) {
          setHidden(dy > 0 && y > 80)
          lastY = y
        }
        raf = null
      })
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (raf !== null) cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <header
      className={cn(
        'sticky top-0 z-20 bg-bg/85 backdrop-blur-md border-b border-border-soft',
        'pt-[var(--sait)]',
        'transition-transform duration-200',
        hidden && '-translate-y-[calc(100%+var(--sait))]',
      )}
    >
      <div
        className={cn(
          'flex items-center gap-3 h-bar',
          'pl-[max(1.25rem,var(--sail))]',
          'pr-[max(1.25rem,var(--sair))]',
        )}
      >
        <Button variant="ghost" size="sm" icon onClick={onBack} title="Quay lại">
          <ChevronLeft size={14} />
        </Button>
        <div className="flex items-center gap-2 text-sm min-w-0 flex-1">
          {title && (
            <span className="text-text-muted truncate">{title}</span>
          )}
          {number && (
            <>
              <span className="text-text-subtle/60">/</span>
              <span className="text-text font-medium tabular shrink-0">
                Ch.{number}
              </span>
            </>
          )}
          <span className="text-text-subtle text-xs ml-auto shrink-0">
            {sourceName} · Raw
          </span>
        </div>
        {total > 0 && (
          <span className="text-xs text-text-subtle tabular shrink-0">
            {total} trang
          </span>
        )}
      </div>
    </header>
  )
}


export const Route = createFileRoute('/raw')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    source:     typeof s.source     === 'string' ? s.source     : '',
    url:        typeof s.url        === 'string' ? s.url        : '',
    title:      typeof s.title      === 'string' ? s.title      : undefined,
    number:     typeof s.number     === 'string' ? s.number     : undefined,
    label:      typeof s.label      === 'string' ? s.label      : undefined,
    materialId: typeof s.materialId === 'number' ? s.materialId : undefined,
    numberNorm: typeof s.numberNorm === 'string' ? s.numberNorm : undefined,
  }),
  component: RawReaderPage,
  staticData: { chrome: 'bare' },
})
