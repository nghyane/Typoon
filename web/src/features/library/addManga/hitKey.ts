// Canonical key for a `SearchHit` row — shared between `ResultsList`
// rendering and pick-tracking state in containers. Lives in its own
// module so React Fast Refresh keeps working in `ResultsList.tsx`
// (the rule only allows component exports per file).

import type { SearchHit } from './fanoutSearch'


export function hitKey(hit: SearchHit): string {
  return `${hit.source.manifest.id}::${hit.manga.id}`
}
