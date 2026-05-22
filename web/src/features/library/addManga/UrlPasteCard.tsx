// UrlPasteCard — auto-resolves a pasted URL to its source detail
// then imports without a confirm step.
//
// Three states:
//   • no match              → "Tạo trống thay" fallback
//   • match + loading       → spinner card while fetchMangaDetail runs
//   • match + fetch error   → error card, user clears the input
//
// On detail-fetch success we forge a SearchHit and call importHit
// directly; the modal closes via the importer's onSuccess.

import { useEffect, useState } from 'react'
import { AlertTriangle, Wand2 } from 'lucide-react'

import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import { Button } from '@shared/ui/Button'
import { Spinner } from '@shared/ui/primitives'

import type { matchSource } from './parseUrl'
import type { ImportToLibrary } from './useImportToLibrary'


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
    <div className="rounded-md bg-warning-bg border border-warning-text/20 px-4 py-3">
      <div className="flex items-start gap-3">
        <AlertTriangle size={14} className="text-warning-text shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-text">Không có nguồn quản lý site này</p>
          <p className="text-xs text-text-subtle mt-1 break-all line-clamp-2">
            {url}
          </p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => importer.importBlank('')}
            disabled={importer.isPending}
            className="mt-3"
          >
            <Wand2 size={14} /> Tạo trống thay
          </Button>
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
        // url + title + cover (the MangaSummary contract). Detail
        // resolution happens inside `importHit` from the resolved
        // `detail` we pass alongside.
        importer.importHit({
          hit: {
            source,
            manga: {
              id:    upstreamRef,
              url:   upstreamRef,
              title: d.title,
              cover: d.cover,
            },
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
      <div className="rounded-md bg-error-bg border border-error-text/20 px-4 py-3">
        <div className="flex items-start gap-3">
          <AlertTriangle size={14} className="text-error-text shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-text">Không tải được từ {manifest.name}</p>
            <p className="text-xs text-error-text mt-1 line-clamp-2">{err}</p>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div className="rounded-md bg-surface-2 px-4 py-3 flex items-center gap-3">
      <Spinner size={14} className="text-info-text shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text">
          {importer.isPending
            ? 'Đang thêm vào thư viện…'
            : `Đang tải từ ${manifest.name}…`}
        </p>
        <p className="text-xs text-text-subtle truncate mt-1">{upstreamRef}</p>
      </div>
    </div>
  )
}
