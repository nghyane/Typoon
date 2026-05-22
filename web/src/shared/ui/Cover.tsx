import { useEffect, useState } from 'react'
import { cn } from '@shared/lib/cn'
import { proxify } from '@features/browse/proxy'

const API_BASE = window.location.hostname.endsWith('.discordsays.com')
  ? ''
  : (import.meta.env.VITE_PUBLIC_BASE_URL ?? '')

// Resolve a cover URL — absolute upstream → CDN proxy,
// relative `/files/...` → API base.
export function coverUrl(src: string | null, version?: string | null): string | null {
  if (!src) return null
  const isAbs = /^https?:\/\//i.test(src)
  const base  = isAbs ? proxify(src) : `${API_BASE}${src}`
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
