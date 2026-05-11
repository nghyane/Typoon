import { useEffect, useState } from 'react'
import { cn } from '@shared/lib/cn'
import { api } from '@shared/api/api'

// Resolve relative API URLs (e.g. "/files/<slug>/cover.jpg") against
// the public base URL when running cross-origin. Same-origin dev/prod
// hits Vite proxy and the relative URL stays untouched. `version` busts
// cache when the upstream cover.jpg gets replaced under the same URL.
export function coverUrl(src: string | null, version?: string | null): string | null {
  if (!src) return null
  const base = /^https?:\/\//i.test(src) ? src : `${api.base}${src}`
  if (!version) return base
  return base.includes('?')
    ? `${base}&v=${encodeURIComponent(version)}`
    : `${base}?v=${encodeURIComponent(version)}`
}

interface Props {
  src:        string | null     // null → fallback ngay
  title:      string
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

  return (
    <div className={cn('flex items-center justify-center overflow-hidden bg-surface-2', className)}>
      {url ? (
        <img
          src={url}
          alt={title}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className={cn('font-black text-text-subtle/60 select-none', fontSize)}>
          {title.slice(0, 2).toUpperCase()}
        </span>
      )}
    </div>
  )
}
