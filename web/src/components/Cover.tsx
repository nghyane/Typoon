import { useEffect, useState } from 'react'
import { cn } from '../lib/cn'
import { api } from '../lib/api'

interface Props {
  src:       string | null   // null → fallback ngay
  title:     string
  className?: string
  fontSize?: string          // tailwind text-* class for fallback letters
}

// Resolve relative API URLs (e.g. "/files/<slug>/cover.jpg") against
// VITE_API_URL when running cross-origin from the API. Same-origin dev/prod
// hits the Vite proxy and the relative URL stays untouched.
function resolve(src: string): string {
  if (!api.base) return src
  if (/^https?:\/\//i.test(src)) return src
  return `${api.base}${src}`
}

export function Cover({ src, title, className, fontSize = 'text-xl' }: Props) {
  const [failed, setFailed] = useState(false)

  // Reset failure state when src changes — without this, switching projects
  // to one with a working cover still shows the fallback.
  useEffect(() => { setFailed(false) }, [src])

  const showImg = src && !failed
  return (
    <div className={cn('flex items-center justify-center overflow-hidden bg-zinc-100', className)}>
      {showImg ? (
        <img
          src={resolve(src)}
          alt={title}
          className="w-full h-full object-cover"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className={cn('font-black text-zinc-300', fontSize)}>
          {title.slice(0, 2).toUpperCase()}
        </span>
      )}
    </div>
  )
}
