import { Link } from '@tanstack/react-router'
import { Cover } from '@shared/ui/Cover'
import { proxify } from '../proxy'
import { useAutoTranslate, shouldTranslate } from '../autoTranslate'
import { useTranslated } from '../useTranslated'
import { isInternal } from '../manifest/internal'
import type { MangaSummary, SourceManifest } from '../manifest/types'

// =============================================================================
// ShelfCard — compact card inside horizontal shelf rails.
//
// Navigation:
//   • External source — links to /browse/$source/manga/$mangaId
//     (the in-app manga detail page).
//   • Internal source (Community) — `manga.url` already encodes the
//     project route (`/projects/123`); we follow it verbatim so the
//     user lands on the existing project detail UI without us
//     re-implementing it inside browse.
// =============================================================================

interface Props {
  source:   string
  manifest: SourceManifest
  manga:    MangaSummary
}

export function ShelfCard({ source, manifest, manga }: Props) {
  const autoEnabled = useAutoTranslate((s) => s.enabled)
  const autoTarget  = useAutoTranslate((s) => s.target)
  const useTr = shouldTranslate(autoEnabled, autoTarget, manifest.languages)
  const trTitle = useTranslated(manga.title, autoTarget, useTr)
  const title = useTr && trTitle ? trTitle : manga.title

  const cardClasses = 'group flex flex-col gap-2 w-[120px] sm:w-[144px] shrink-0'
  const body = (
    <>
      <Cover
        src={manga.cover ? proxify(manga.cover) : null}
        title={title}
        className="w-full aspect-[2/3] rounded-md group-hover:brightness-110 transition-[filter]"
      />
      <p className="text-[13px] font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
        {title}
      </p>
    </>
  )

  if (isInternal(manifest)) {
    return (
      <Link to={manga.url as never} className={cardClasses}>
        {body}
      </Link>
    )
  }
  return (
    <Link
      to="/browse/$source/manga/$mangaId"
      params={{ source, mangaId: encodeURIComponent(manga.id) }}
      className={cardClasses}
    >
      {body}
    </Link>
  )
}
