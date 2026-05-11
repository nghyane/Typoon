import { Link } from '@tanstack/react-router'
import { Cover } from '@shared/ui/Cover'
import { proxify } from '../proxy'
import { isInternal } from '../manifest/internal'
import type { MangaSummary, SourceManifest } from '../manifest/types'

// =============================================================================
// MangaCard — single tile in /browse/$source feed grid.
//
// Cover proxied through bunle-cdn (Referer/UA injected per host).
//
// When `translatedTitle` is provided (auto-translate enabled and the
// source language ≠ user target), it replaces the displayed title;
// the original is shown small below for context. While the
// translation is pending the original title stays — no flicker.
//
// Navigation matches ShelfCard:
//   • External source → /browse/$source/manga/$mangaId
//   • Internal (community) → follows `manga.url` verbatim (project route)
// =============================================================================

interface Props {
  source:           string
  manifest:         SourceManifest
  manga:            MangaSummary
  translatedTitle?: string | null
}

export function MangaCard({ source, manifest, manga, translatedTitle }: Props) {
  const showTr = translatedTitle && translatedTitle !== manga.title
  const display = showTr ? translatedTitle! : manga.title

  const body = (
    <>
      <Cover
        src={manga.cover ? proxify(manga.cover) : null}
        title={display}
        className="w-full aspect-[2/3] rounded-md mb-2.5 group-hover:brightness-110 transition-[filter]"
      />
      <p className="text-sm font-medium text-text leading-snug line-clamp-2 group-hover:text-accent-text transition-colors">
        {display}
      </p>
      {showTr && (
        <p className="text-[10px] text-text-subtle leading-tight line-clamp-1 mt-0.5 italic">
          {manga.title}
        </p>
      )}
    </>
  )

  if (isInternal(manifest)) {
    return (
      <Link to={manga.url as never} className="group block">
        {body}
      </Link>
    )
  }
  return (
    <Link
      to="/browse/$source/manga/$mangaId"
      params={{ source, mangaId: encodeURIComponent(manga.id) }}
      className="group block"
    >
      {body}
    </Link>
  )
}
