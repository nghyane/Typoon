// Shared types for AddMangaModal subcomponents.

import type { InstalledSource } from '@features/browse/manifest/types'

/** A manga successfully resolved (either from URL paste or search
 *  pick). Carries everything material.import needs to dedupe + the
 *  display snapshot the library card will show. */
export interface Picked {
  source:      InstalledSource
  upstreamRef: string
  title:       string
  cover:       string | null
  description: string | null
  author:      string | null
  status:      string | null
  languages:   string[]
  nsfw:        boolean
}

export type Mode = 'search' | 'picked' | 'manual'
