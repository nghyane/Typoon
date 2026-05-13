import { useEffect, useState } from 'react'
import { AlertTriangle, Loader2, Wand2 } from 'lucide-react'
import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import type { matchSource } from './parseUrl'
import type { Picked } from './types'

// =============================================================================
// UrlPasteCard — three states for a URL paste:
//
//   • match + loading  spinner + 'Đang tải từ {source}…'
//   • match + error    error card with the source name + reason
//   • no match         warning card with a 'Tạo thủ công' CTA
//
// On loading success, we immediately call onPick so the parent flips
// to PickedDetail. The card unmounts in the same render, so 'loading'
// is the only state the user sees until the network call returns.
// =============================================================================

export function UrlPasteCard({
  url, match, onPick, onManualCreate,
}: {
  url:            string
  match:          ReturnType<typeof matchSource>
  onPick:         (p: Picked) => void
  onManualCreate: (seed: string) => void
}) {
  if (!match) {
    return <UnsupportedUrlCard url={url} onManualCreate={onManualCreate} />
  }
  return <MatchedUrlCard match={match} onPick={onPick} />
}


function UnsupportedUrlCard({
  url, onManualCreate,
}: {
  url: string; onManualCreate: (seed: string) => void
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
            onClick={() => onManualCreate('')}
            className="mt-2.5 inline-flex items-center gap-2 h-7 px-2.5 rounded-sm bg-surface-2 text-xs text-text hover:bg-hover cursor-pointer transition-colors"
          >
            <Wand2 size={12} />
            Tạo thủ công thay
          </button>
        </div>
      </div>
    </div>
  )
}


function MatchedUrlCard({
  match, onPick,
}: {
  match: NonNullable<ReturnType<typeof matchSource>>
  onPick: (p: Picked) => void
}) {
  const { source, upstreamRef } = match
  const manifest = source.manifest
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setErr(null)
    fetchMangaDetail(manifest, upstreamRef)
      .then((d) => {
        if (cancelled) return
        onPick({
          source, upstreamRef,
          title:       d.title,
          cover:       d.cover,
          description: d.description,
          author:      d.author,
          status:      d.status,
          languages:   d.availableLanguages ?? manifest.languages,
          nsfw:        !!manifest.nsfw,
        })
      })
      .catch((e) => {
        if (cancelled) return
        setErr(e instanceof Error ? e.message : 'Không tải được trang truyện')
      })
    return () => { cancelled = true }
  }, [manifest, upstreamRef, source, onPick])

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
        <p className="text-sm text-text">Đang tải từ {manifest.name}…</p>
        <p className="text-xs text-text-subtle truncate mt-0.5">{upstreamRef}</p>
      </div>
    </div>
  )
}
