import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { SearchResults } from '@features/browse/views/SearchResults'
import { useResolvedSource } from '@features/browse/useResolvedSource'

interface SearchParams { q?: string }

function SearchPage() {
  const { q = '' } = Route.useSearch()
  const source = useResolvedSource()
  const nav = useNavigate()
  if (!source) return null
  return (
    <SearchResults
      source={source}
      initialQ={q}
      onQueryChange={(next) =>
        nav({
          to: '/browse/$source/search',
          params: { source: source.manifest.id },
          search: { q: next || undefined } as never,
        })
      }
    />
  )
}

export const Route = createFileRoute('/browse/$source/search')({
  validateSearch: (s: Record<string, unknown>): SearchParams => ({
    q: typeof s.q === 'string' && s.q.length > 0 ? s.q : undefined,
  }),
  component: SearchPage,
})
