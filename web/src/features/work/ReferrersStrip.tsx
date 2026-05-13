// ReferrersStrip — surface a Work's `cross_refs` as clickable links
// out to the external identity services (Anilist, MAL, MangaDex, …).
//
// Lookup order for each namespace:
//
//   1. Bundled link plugin (`packages/link-plugins/`) — declares
//      `url_template` so any new plugin auto-renders its referrer.
//   2. Static fallback for source-manifest namespaces (mangadex,
//      bato, …) that don't have a link plugin yet but still publish
//      cross_refs from the manifest layer.
//
// Hidden entirely when the Work has no cross_refs. The strip is
// purely informational — clicks open the external page in a new tab
// so the viewer can verify identity before voting in the
// LinkSuggestionPanel.

import { ExternalLink } from 'lucide-react'

import { bundledLinkPlugins } from '@features/link/plugins'
import { cn } from '@shared/lib/cn'


/** Static fallbacks for namespaces that don't (yet) ship as a link
 *  plugin but the manifest layer still emits. Plain map, namespace
 *  → display name + url template. */
const FALLBACK: Record<string, { name: string; urlTemplate: string }> = {
  mdex_uuid: {
    name:        'MangaDex',
    urlTemplate: 'https://mangadex.org/title/{id}',
  },
  mu: {
    name:        'MangaUpdates',
    urlTemplate: 'https://www.mangaupdates.com/series.html?id={id}',
  },
  mal: {
    name:        'MyAnimeList',
    urlTemplate: 'https://myanimelist.net/manga/{id}',
  },
  kitsu: {
    name:        'Kitsu',
    urlTemplate: 'https://kitsu.io/manga/{id}',
  },
}


interface Props {
  crossRefs: Record<string, unknown> | null | undefined
}


export function ReferrersStrip({ crossRefs }: Props) {
  const items = resolve(crossRefs)
  if (items.length === 0) return null

  return (
    <div className="flex items-center gap-2 flex-wrap text-xs text-text-muted">
      <span className="text-text-subtle shrink-0">Tham chiếu:</span>
      {items.map((it) => (
        <a
          key={it.namespace}
          href={it.url}
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            'inline-flex items-center gap-1 h-6 px-2 rounded-sm',
            'bg-surface-2 text-text hover:bg-hover transition-colors',
          )}
          title={`Mở ${it.name} ở tab mới`}
        >
          <span>{it.name}</span>
          <ExternalLink size={11} className="opacity-70" />
        </a>
      ))}
    </div>
  )
}


// ── Resolver ───────────────────────────────────────────────────


interface ResolvedRef {
  namespace: string
  name:      string
  url:       string
}


function resolve(
  refs: Record<string, unknown> | null | undefined,
): ResolvedRef[] {
  if (!refs) return []
  const out: ResolvedRef[] = []
  for (const [ns, raw] of Object.entries(refs)) {
    const id = stringifyId(raw)
    if (!id) continue
    const tpl  = templateFor(ns)
    if (!tpl) continue
    out.push({
      namespace: ns,
      name:      tpl.name,
      url:       tpl.urlTemplate.replace('{id}', encodeURIComponent(id)),
    })
  }
  // Stable order: link-plugin order first, then fallback alphabetical.
  // Keeps the strip deterministic across renders.
  return out.sort((a, b) => a.namespace.localeCompare(b.namespace))
}


function templateFor(namespace: string): { name: string; urlTemplate: string } | null {
  const plugin = bundledLinkPlugins.find((p) => p.namespace === namespace)
  if (plugin?.url_template) {
    return { name: plugin.name, urlTemplate: plugin.url_template }
  }
  return FALLBACK[namespace] ?? null
}


function stringifyId(v: unknown): string | null {
  if (v == null) return null
  if (typeof v === 'string') return v.trim() || null
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  return null
}
