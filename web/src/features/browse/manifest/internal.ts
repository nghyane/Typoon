// Internal source adapter — wraps typoon's own /api/projects endpoint
// so the Community source slots into the same browse hub as external
// manga sites.
//
// We branch at the UI layer (BrowseSourceHome, ShelfDetail, source
// row click) rather than coercing this into a fake BrowseEndpoint.
// The shelves and chapter list don't share any HTTP shape with the
// generic runtime — pretending they do would add casts that hide
// real divergence.
//
// Contract for callers:
//   • `isInternal(manifest)` — runtime check
//   • `internalShelves(manifest)` — static shelf metadata
//   • `fetchInternalBrowse(shelfId, { page })` — returns MangaSummary[]
//   • `projectToSummary(p)` — used when a project must be rendered
//                             as a card outside the shelf flow

import { api, type ApiProject, type ProjectFilter } from '@shared/api/api'
import { coverUrl } from '@shared/ui/Cover'
import type { MangaSummary, SourceManifest } from './types'

export function isInternal(manifest: SourceManifest): boolean {
  return manifest.kind === 'internal'
}

interface InternalShelf {
  id:     string
  label:  string
  hint?:  string
  filter: ProjectFilter
}
export type { InternalShelf }

const COMMUNITY_SHELVES: InternalShelf[] = [
  { id: 'community', label: 'Mới chia sẻ',    filter: 'community' },
  { id: 'pinned',    label: 'Đã lưu',         filter: 'pinned' },
  { id: 'mine',      label: 'Truyện của bạn', filter: 'mine' },
]

/** Static shelf metadata for an internal source. UI uses this list
 *  directly — never goes through `manifest.endpoints.shelves`. */
export function internalShelves(manifest: SourceManifest): InternalShelf[] {
  if (manifest.id !== 'community') return []
  return COMMUNITY_SHELVES
}

/** Convert an ApiProject → MangaSummary so rail views render the
 *  same MangaCard / ShelfCard primitive without per-source branching.
 *  `id` and `url` use the project route — the routing layer
 *  recognises this shape and skips MangaPage. */
export function projectToSummary(p: ApiProject): MangaSummary {
  return {
    id:    `/projects/${p.project_id}`,
    url:   `/projects/${p.project_id}`,
    title: p.title,
    cover: coverUrl(p.cover_url, p.updated_at),
  }
}

/** Fetch one shelf. Pagination follows the same page-based shape as
 *  external sources; we cap client-side for parity with rail UX. */
export async function fetchInternalBrowse(
  shelfId: string,
  args: { page?: number } = {},
): Promise<MangaSummary[]> {
  const conf = COMMUNITY_SHELVES.find((s) => s.id === shelfId)
  if (!conf) return []
  const projects = await api.listProjects(conf.filter)
  const page = args.page ?? 1
  const PER  = 24
  const start = (page - 1) * PER
  return projects.slice(start, start + PER).map(projectToSummary)
}
