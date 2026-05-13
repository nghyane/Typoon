// Link plugin runtime — declarative cross-reference lookup for the
// Work auto-enrich flow.
//
// A plugin (e.g. `packages/link-plugins/anilist.json`) describes a
// 3rd-party identity service (Anilist, MAL, MangaUpdates, …):
//
//   • how to hit its search endpoint (URL + method + body template),
//   • the JSONPath that extracts the result list,
//   • the JSONPath of each field on a row (id, title, native title,
//     synonyms, cover).
//
// The runtime fans the lookup out across every installed plugin in
// parallel, collects normalized candidates, and hands them back to
// `useAutoEnrichWork` which decides what to do with them.
//
// We deliberately keep this self-contained — no shared code with
// `features/browse/manifest/runtime.ts` because:
//   - the manga-source runtime parses HTML too (not relevant here);
//   - link plugins use POST-JSON bodies with templated variables,
//     a shape the manifest runtime doesn't expose.
// Two small focused runtimes are easier to evolve than one giant
// shared one.

import { proxify } from '@features/browse/proxy'


export interface LinkPlugin {
  id:          string
  name:        string
  namespace:   string
  description?: string
  /** URL template the SPA uses to deep-link to the external entry,
   *  e.g. `"https://anilist.co/manga/{id}"`. `{id}` is substituted
   *  with the value in `cross_refs[namespace]`. Optional — without
   *  it, the Referrers UI skips the link. */
  url_template?: string
  endpoints: {
    search: SearchEndpoint
  }
}

interface SearchEndpoint {
  url:      string
  method:   'GET' | 'POST'
  headers?: Record<string, string>
  /** JSON body with `{var}` placeholders. We substitute variables
   *  client-side; null when GET. */
  bodyJson?: unknown
  /** Dotted JSONPath to the result list, e.g. "data.Page.media". */
  list:     string
  /** Each entry maps an output field name to a dotted path INSIDE
   *  one row, e.g. `"title": "title.romaji"`. Use `$.x` style for
   *  symmetry with the manifest runtime, but plain `x.y.z` works too. */
  fields:   Record<string, string>
}


/** One result row, normalized across plugins. `id` is whatever the
 *  service uses (string or number, depends on plugin); the caller
 *  stores it under `cross_refs[plugin.namespace]`. */
export interface LinkCandidate {
  plugin:       string             // plugin.id
  namespace:    string             // plugin.namespace
  externalId:   string             // service-specific id (stringified)
  title:        string | null
  titleNative:  string | null
  titleAlt:     string[]
  cover:        string | null
}


/** Query shape — caller passes whatever title strings they have. The
 *  runtime picks the best one per plugin (some prefer native, others
 *  romanized). For Anilist's `search` arg either works; we send the
 *  native title when present because kanji is a stronger signal. */
export interface LinkQuery {
  title:       string
  titleNative?: string | null
}


/** Fanout across every plugin in parallel. Each plugin gets its own
 *  network call; failures (network, parse error, rate limit) are
 *  swallowed so one broken plugin doesn't sink the whole enrich
 *  attempt. Returns a flat list of candidates — multiple per plugin
 *  is fine, the caller dedupes by `(plugin, externalId)`. */
export async function lookupAcrossPlugins(
  plugins: LinkPlugin[],
  query:   LinkQuery,
  opts:    { signal?: AbortSignal } = {},
): Promise<LinkCandidate[]> {
  const q = (query.titleNative?.trim() || query.title.trim())
  if (!q) return []

  const results = await Promise.allSettled(
    plugins.map((p) => lookupOne(p, q, opts.signal)),
  )

  const out: LinkCandidate[] = []
  for (const r of results) {
    if (r.status === 'fulfilled') out.push(...r.value)
  }
  return out
}


// ── Internals ──────────────────────────────────────────────────


async function lookupOne(
  plugin: LinkPlugin,
  q:      string,
  signal: AbortSignal | undefined,
): Promise<LinkCandidate[]> {
  const ep = plugin.endpoints.search
  const init: RequestInit = {
    method:  ep.method,
    headers: ep.headers,
    signal,
  }
  if (ep.method === 'POST' && ep.bodyJson != null) {
    init.body = JSON.stringify(substitute(ep.bodyJson, { q }))
  }
  // Route through `/proxy` so the browser doesn't hit CORS on
  // services that don't ship permissive headers. Anilist allows
  // CORS, but routing every plugin through the same proxy keeps
  // the plugin definitions free of "does this need a proxy?" flags.
  const res = await fetch(proxify(ep.url), init)
  if (!res.ok) return []
  const json = await res.json()

  const list = getPath(json, ep.list)
  if (!Array.isArray(list)) return []

  return list.map((row): LinkCandidate => {
    const f = (key: string) => getPath(row, ep.fields[key] ?? '')
    const rawId = f('id')
    const idStr = rawId == null ? '' : String(rawId)
    return {
      plugin:      plugin.id,
      namespace:   plugin.namespace,
      externalId:  idStr,
      title:       toStr(f('title')),
      titleNative: toStr(f('title_native')),
      titleAlt:    toStrArray(f('title_alt'), f('synonyms')),
      cover:       toStr(f('cover')),
    }
  }).filter((c) => c.externalId !== '')
}


/** Dotted-path JSON lookup. Accepts `$.a.b` or `a.b`; arrays use
 *  numeric indices. Returns undefined when any segment misses. */
function getPath(obj: unknown, path: string): unknown {
  if (!path) return undefined
  const segs = path.replace(/^\$\.?/, '').split('.').filter(Boolean)
  let cur: unknown = obj
  for (const seg of segs) {
    if (cur == null) return undefined
    if (Array.isArray(cur)) {
      const idx = Number(seg)
      cur = Number.isInteger(idx) ? cur[idx] : undefined
    } else if (typeof cur === 'object') {
      cur = (cur as Record<string, unknown>)[seg]
    } else {
      return undefined
    }
  }
  return cur
}


/** Recursively substitute `{q}` style placeholders in a JSON
 *  structure, returning a deep copy with strings replaced. */
function substitute(value: unknown, vars: Record<string, string>): unknown {
  if (typeof value === 'string') {
    return value.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? '')
  }
  if (Array.isArray(value)) return value.map((v) => substitute(v, vars))
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(value)) out[k] = substitute(v, vars)
    return out
  }
  return value
}


function toStr(v: unknown): string | null {
  if (v == null) return null
  if (typeof v === 'string') return v.trim() || null
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  return null
}


function toStrArray(...vals: unknown[]): string[] {
  const out: string[] = []
  for (const v of vals) {
    if (v == null) continue
    if (Array.isArray(v)) {
      for (const item of v) {
        const s = toStr(item)
        if (s) out.push(s)
      }
    } else {
      const s = toStr(v)
      if (s) out.push(s)
    }
  }
  return out
}
