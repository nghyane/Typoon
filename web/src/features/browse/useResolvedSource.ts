// Hook shared by `/browse/$source/*` child routes to resolve the
// current source manifest. Lives in its own module so the route files
// stay component-only (React Refresh happy).

import { useParams } from '@tanstack/react-router'
import { useSources, getSource } from '@features/browse/sources'
import type { InstalledSource } from '@features/browse/manifest/types'

export function useResolvedSource(): InstalledSource | null {
  // `strict: false` — the hook is callable from any descendant of
  // /browse/$source without each consumer asserting the exact route.
  const params = useParams({ strict: false }) as { source?: string }
  const id = params.source
  const installed = useSources((s) => (id ? s.sources[id] : undefined))
  if (!id) return null
  return installed ?? getSource(id)
}
