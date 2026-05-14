import { useEffect, useState } from 'react'
import { cn } from '@shared/lib/cn'
import { api } from '@shared/api/api'
import { proxify } from '@features/browse/proxy'

// Resolve a cover URL into something the browser can actually load:
//
//   • Absolute upstream URL (e.g. `https://uploads.mangadex.org/...`)
//     → rewritten through the CDN proxy. Upstream hosts hotlink-block
//     or lack CORS, and the server stores raw upstream URLs on
//     materials/library entries (no local cover storage yet), so the
//     proxy is the only path that consistently renders.
//
//   • Relative API path (e.g. `/files/<slug>/cover.jpg`) → resolved
//     against `api.base` for cross-origin deployments. Same-origin
//     dev/prod hits the Vite proxy and the relative URL stays
//     untouched.
//
// `version` busts the browser cache when the upstream cover gets
// replaced under the same URL (e.g. after re-enrich).
export function coverUrl(src: string | null, version?: string | null): string | null {
  if (!src) return null
  const isAbs = /^https?:\/\//i.test(src)
  const base  = isAbs ? proxify(src) : `${api.base}${src}`
  if (!version) return base
  return base.includes('?')
    ? `${base}&v=${encodeURIComponent(version)}`
    : `${base}?v=${encodeURIComponent(version)}`
}

interface Props {
  src:        string | null     // null → fallback ngay
  title:      string | null | undefined
  className?: string
  fontSize?:  string             // tailwind text-* class for fallback letters
  version?:   string | null
}

export function Cover({ src, title, className, fontSize = 'text-xl', version }: Props) {
  const [failed, setFailed] = useState(false)

  // Reset failure state when src changes — without this, switching projects
  // to one with a working cover still shows the fallback.
  useEffect(() => { setFailed(false) }, [src])

  const url = !failed ? coverUrl(src, version) : null
  const safeTitle = (title ?? '').trim()

  return (
    <div className={cn('flex items-center justify-center overflow-hidden bg-surface-2', className)}>
      {url ? (
        <img
          src={url}
          alt={safeTitle}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className={cn('font-black text-text-subtle/60 select-none', fontSize)}>
          {safeTitle.slice(0, 2).toUpperCase() || '—'}
        </span>
      )}
    </div>
  )
}
