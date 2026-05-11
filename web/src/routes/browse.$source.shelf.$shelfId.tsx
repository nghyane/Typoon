import { createFileRoute } from '@tanstack/react-router'
import { useState } from 'react'
import { ShelfDetail } from '@features/browse/views/ShelfDetail'
import type { FilterState } from '@features/browse/views/FilterBar'
import type { SourceManifest } from '@features/browse/manifest/types'
import { useResolvedSource } from '@features/browse/useResolvedSource'

function ShelfDetailPage() {
  const { shelfId } = Route.useParams()
  const source = useResolvedSource()
  const [filters, setFilters] = useState<FilterState>(
    () => defaultFilterState(source?.manifest),
  )
  if (!source) return null
  return (
    <ShelfDetail
      source={source}
      shelfId={shelfId}
      filters={filters}
      onFiltersChange={setFilters}
    />
  )
}

function defaultFilterState(manifest: SourceManifest | undefined): FilterState {
  const out: FilterState = {}
  if (!manifest?.filters) return out
  for (const f of manifest.filters) {
    const d = manifest.defaults?.[f.id]
    if (d !== undefined) {
      out[f.id] = d
    } else {
      out[f.id] = f.type === 'multi' ? [] : (f.options[0]?.id ?? '')
    }
  }
  return out
}

export const Route = createFileRoute('/browse/$source/shelf/$shelfId')({
  component: ShelfDetailPage,
})
