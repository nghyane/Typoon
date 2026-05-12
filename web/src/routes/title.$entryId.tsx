import { createFileRoute, redirect } from '@tanstack/react-router'
import { TitleHub } from '@features/title/TitleHub'

// URL is the state of truth for filter / search / sort, mirroring the
// pre-refactor /projects/$projectId route. Refresh keeps the user's
// filter; sharing a link captures the query.

export type StatusFilter = 'all' | 'translated' | 'running' | 'error' | 'raw'
export type Sort         = 'chapter_desc' | 'chapter_asc' | 'updated_desc'

const FILTERS: ReadonlyArray<StatusFilter> = [
  'all', 'translated', 'running', 'error', 'raw',
]
const SORTS: ReadonlyArray<Sort> = [
  'chapter_desc', 'chapter_asc', 'updated_desc',
]

interface SearchParams {
  filter?: StatusFilter
  q?:      string
  sort?:   Sort
}

function TitleHubPage() {
  const { entryId } = Route.useParams()
  const id = Number(entryId)
  return <TitleHub entryId={id} />
}

export const Route = createFileRoute('/title/$entryId')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    filter: FILTERS.includes(s.filter as StatusFilter) ? (s.filter as StatusFilter) : undefined,
    q:      typeof s.q === 'string' && s.q.length > 0 ? s.q : undefined,
    sort:   SORTS.includes(s.sort as Sort) ? (s.sort as Sort) : undefined,
  }),
  beforeLoad: ({ params }) => {
    const id = Number(params.entryId)
    if (!Number.isFinite(id) || id <= 0) {
      throw redirect({ to: '/library' })
    }
  },
  component: TitleHubPage,
})
