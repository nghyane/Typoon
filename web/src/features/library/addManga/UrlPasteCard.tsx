import { useEffect, useState } from 'react'
import { AlertTriangle, Loader2, Wand2 } from 'lucide-react'

import { fetchMangaDetail } from '@features/browse/manifest/runtime'

import type { matchSource } from './parseUrl'
import type { ImportToLibrary } from './useImportToLibrary'

// Three states for a URL paste: match+loading, match+error, no-match.
// On detail-fetch success we forge a SearchHit and call importHit
// directly; the modal closes via the importer's onSuccess. No
// confirm step.

export function UrlPasteCard({
  url, match, importer,
}: {
  url:      string
  match:    ReturnType<typeof matchSource>
  importer: ImportToLibrary
}) {
  if (!match) return <UnsupportedUrlCard url={url} importer={importer} />
  return <MatchedUrlCard match={match} importer={importer} />
}


function UnsupportedUrlCard({
  url, importer,
}: {
  url:      string
  importer: ImportToLibrary
}) {
  return (
    <div className="rounded-md bg-warning/10 border border-warning/20 px-4 py-3">
      <div className="flex items-start gap-2.5">
        <AlertTriangle size={14} className="text-warning-text shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Không có nguồn quản lý site này</p>
          <p className="text-xs text-text-subtle mt-0.5 break-all line-clamp-2">
            {url}
          </p>
          <button
            type="button"
            onClick={() => importer.importBlank('')}
            disabled={importer.isPending}
            className="mt-2.5 inline-flex items-center gap-2 h-7 px-2.5 rounded-sm bg-surface-2 text-xs text-text hover:bg-hover cursor-pointer transition-colors disabled:cursor-wait disabled:opacity-60"
          >
            <Wand2 size={12} />
            Tạo trống thay
          </button>
        </div>
      </div>
    </div>
  )
}


function MatchedUrlCard({
  match, importer,
}: {
  match:    NonNullable<ReturnType<typeof matchSource>>
  importer: ImportToLibrary
}) {
  const { source, upstreamRef } = match
  const manifest = source.manifest
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const ctrl = new AbortController()
    setErr(null)
    fetchMangaDetail(manifest, upstreamRef)
      .then((d) => {
        if (ctrl.signal.aborted) return
        // Forge a SearchHit from the URL match so the same importer
        // path covers both flows. The "manga snapshot" carries only
        // url + title + cover (the MangaSummary contract). All
        // language / status / author resolution happens inside
        // `importHit` from the resolved `detail` we pass alongside.
        importer.importHit({
          hit: {
            source,
            manga: {
              id:    upstreamRef,
              url:   upstreamRef,
              title: d.title,
              cover: d.cover,
            },
            // Score doesn't apply to URL-paste flow (we already
            // resolved the canonical row); use 1.0 so any consumer
            // that reads it sees a "100% match" rather than 0.
            score: 1,
          },
          detail: d,
        })
      })
      .catch((e) => {
        if (ctrl.signal.aborted) return
        setErr(e instanceof Error ? e.message : 'Không tải được trang truyện')
      })
    return () => ctrl.abort()
  }, [manifest, upstreamRef, source, importer])

  if (err) {
    return (
      <div className="rounded-md bg-error/10 border border-error/20 px-4 py-3">
        <div className="flex items-start gap-2.5">
          <AlertTriangle size={14} className="text-error-text shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-text">Không tải được từ {manifest.name}</p>
            <p className="text-xs text-error-text mt-0.5 line-clamp-2">{err}</p>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="rounded-md bg-surface-2 px-4 py-3 flex items-center gap-2.5">
      <Loader2 size={14} className="text-info-text animate-spin shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text">
          {importer.isPending
            ? 'Đang thêm vào thư viện…'
            : `Đang tải từ ${manifest.name}…`}
        </p>
        <p className="text-xs text-text-subtle truncate mt-0.5">{upstreamRef}</p>
      </div>
    </div>
  )
}
