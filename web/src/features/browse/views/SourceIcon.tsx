import { useState } from 'react'
import { cn } from '@shared/lib/cn'
import { proxify } from '../proxy'
import type { SourceManifest } from '../manifest/types'

// =============================================================================
// SourceIcon — neutral monogram tile.
//
// Design intent (Linear / Vercel / Notion / Plane pattern):
//   • Backdrop is one neutral surface across every source — never a
//     coloured tint. Saturated icon tiles read as toy / kid-shop UI
//     and clash with the photo-heavy content (manga covers).
//   • Two characters of the source name, drawn in `text-muted` —
//     the icon is a quiet landmark, not an attention magnet.
//   • A real image (manifest.icon) overrides the monogram. Falls back
//     when the image fails to load.
//
// `accent` from the manifest is intentionally ignored for the tile
// itself. It still drives small hover text affordances elsewhere
// (kept on the manifest for future use without re-migration).
// =============================================================================

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  if (parts.length === 0) return '?'
  if (parts.length === 1) {
    const w = parts[0]!
    if (w.length >= 2) {
      const mid = midCapital(w)
      if (mid) return (w[0]! + mid).toUpperCase()
      return (w[0]! + w[1]!).toUpperCase()
    }
    return w[0]!.toUpperCase()
  }
  return (parts[0]![0]! + parts[1]![0]!).toUpperCase()
}

function midCapital(word: string): string | null {
  for (let i = 1; i < word.length; i++) {
    const c = word[i]!
    if (c >= 'A' && c <= 'Z') return c
  }
  return null
}

interface Props {
  manifest:   SourceManifest
  className?: string
  /** Tailwind font-size class. Default tuned for size-12 (48px). */
  fontSize?:  string
}

export function SourceIcon({ manifest, className, fontSize = 'text-[13px]' }: Props) {
  const [imgFailed, setImgFailed] = useState(false)
  const useImg = !!manifest.icon && !imgFailed
  const label  = initials(manifest.name)

  return (
    <div
      className={cn(
        'relative grid place-items-center overflow-hidden shrink-0',
        'rounded-md select-none',
        'bg-surface-2',
        // 1px inset ring gives depth without a border on the layout grid.
        'ring-1 ring-inset ring-border-soft',
        className,
      )}
      aria-label={manifest.name}
    >
      {useImg ? (
        <img
          src={proxify(manifest.icon!)}
          alt=""
          className="w-full h-full object-cover"
          onError={() => setImgFailed(true)}
        />
      ) : (
        <span
          className={cn(
            'font-semibold tracking-tight tabular leading-none text-text-muted',
            fontSize,
          )}
        >
          {label}
        </span>
      )}
    </div>
  )
}
