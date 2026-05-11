import { Link } from '@tanstack/react-router'
import { Cover } from '@shared/ui/Cover'
import { proxify } from '../proxy'
import { useAutoTranslate, shouldTranslate } from '../autoTranslate'
import { useTranslated } from '../useTranslated'
import type { MangaSummary, SourceManifest } from '../manifest/types'

// =============================================================================
// ShelfCard — compact card inside horizontal shelf rails.
//
// All cards land on /browse/$source/manga/$mangaId regardless of
// origin. The detail page handles import/translate spawn flow.
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

  return (
    <Link
      to="/browse/$source/manga/$mangaId"
      params={{ source, mangaId: encodeURIComponent(manga.id) }}
      className="group flex flex-col gap-2 w-[120px] sm:w-[144px] shrink-0"
    >
      <Cover
        src={manga.cover ? proxify(manga.cover) : null}
        title={title}
        className="w-full aspect-[2/3] rounded-md group-hover:brightness-110 transition-[filter]"
      />
      <p className="text-[13px] font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
        {title}
      </p>
    </Link>
  )
}
