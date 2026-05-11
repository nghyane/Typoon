import { Link } from '@tanstack/react-router'
import { Cover } from '@shared/ui/Cover'
import { proxify } from '../proxy'
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
// Navigation: every card opens /browse/$source/manga/$mangaId. The
// detail page (BrowseMangaPage) is the gateway into the material
// flow; if the user wants their library entry's primary material,
// they click through from /library.
// =============================================================================

interface Props {
  source:           string
  // Reserved for callers that want a per-source visual treatment
  // (NSFW chip, lang flag); kept on Props so existing rails don't
  // need to be touched in this slice.
  manifest:         SourceManifest
  manga:            MangaSummary
  translatedTitle?: string | null
}

export function MangaCard({ source, manga, translatedTitle }: Props) {
  const showTr = translatedTitle && translatedTitle !== manga.title
  const display = showTr ? translatedTitle! : manga.title

  return (
    <Link
      to="/browse/$source/manga/$mangaId"
      params={{ source, mangaId: encodeURIComponent(manga.id) }}
      className="group block"
    >
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
    </Link>
  )
}
