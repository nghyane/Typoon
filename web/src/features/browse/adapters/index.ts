// Adapter registry — maps adapter id (manifest.adapter field) to
// the SourceAdapter implementation.
//
// Adding a new site:
//   1. Create adapters/{site}.ts implementing SourceAdapter.
//   2. Import and register below.
//   3. Set `"adapter": "{site}"` in the source manifest JSON.

import type { SourceAdapter } from './types'
import { hentaifoxAdapter } from './hentaifox'
import { ehentaiAdapter }   from './ehentai'
import { hitomiAdapter }    from './hitomi'

const REGISTRY: Record<string, SourceAdapter> = {
  hentaifox: hentaifoxAdapter,
  ehentai:   ehentaiAdapter,
  hitomi:    hitomiAdapter,
}

export function getAdapter(id: string): SourceAdapter | null {
  return REGISTRY[id] ?? null
}
