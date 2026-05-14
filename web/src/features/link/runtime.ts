// Link plugin runtime — cross-reference lookup for the Work
// auto-enrich flow.
//
// A plugin is a TypeScript adapter — built-in, not user-installable —
// that knows how to talk to one 3rd-party identity service (Anilist,
// MangaDex, MAL, MangaUpdates, …) and how to map its native response
// shape onto a normalized `LinkCandidate`. Adapters live under
// `features/link/plugins/` and are wired into `bundledLinkPlugins`.
//
// We deliberately keep this self-contained — no shared code with the
// manga-source manifest runtime. Link plugins handle wildly different
// response shapes (Anilist GraphQL vs MangaDex JSON:API with nested
// relationships); a declarative JSONPath language fits neither
// comfortably, and the JSON-based plugin schema is dead weight when
// every plugin is built-in.

import { proxify } from '@features/browse/proxy'


/** One result row, normalized across plugins. `externalId` is
 *  whatever the service uses (string or number, depends on plugin);
 *  the caller stores it under `cross_refs[plugin.namespace]`. */
export interface LinkCandidate {
  plugin:       string             // plugin.id
  namespace:    string             // plugin.namespace, used as cross_refs key
  externalId:   string             // service-specific id (stringified)
  title:        string | null
  titleEnglish: string | null
  titleNative:  string | null
  /** Lang-tagged display titles the plugin can attribute to a
   *  specific BCP-47 code. Aggregators (MangaBaka, MangaDex) carry
   *  multi-lang official titles here; the enrich pipeline merges
   *  this map straight into `materials.title_locale` so the title
   *  resolver finds the right name when the viewer's
   *  `preferred_target_lang` matches one of the keys.
   *
   *  Values that already appear as `titleEnglish` / `titleNative`
   *  are deduped server-side; plugins may freely include them. */
  titleLocale:  Record<string, string>
  synonyms:     string[]
  cover:        string | null
  startYear:    number | null
}


/** Query shape — caller passes whatever title strings they have.
 *  Each adapter picks the best one for its endpoint (some prefer
 *  native script for accuracy, others romanized). */
export interface LinkQuery {
  title:       string
  titleNative?: string | null
}


/** Adapter contract every built-in plugin implements. The shape is
 *  intentionally small: identity + a single async `search` method.
 *  Per-plugin transport / retry / rate-limit logic lives inside the
 *  adapter — the runtime only fans out and collects results. */
export interface LinkPluginAdapter {
  id:          string
  namespace:   string
  name:        string
  description?: string
  /** Run a search and return normalized candidates. Failures
   *  (network, parse error, upstream outage) MUST resolve to `[]`,
   *  never throw — the fanout caller relies on per-plugin failures
   *  being silent so one dead service doesn't sink the enrich. */
  search:      (q: LinkQuery, signal?: AbortSignal) => Promise<LinkCandidate[]>
}


// Backwards-compatible alias for call sites that imported the old
// schema-based type. The shape is different (adapter, not config),
// but the role is identical.
export type LinkPlugin = LinkPluginAdapter


/** Fanout across every plugin in parallel. Each plugin gets its own
 *  network call; failures (network, parse error, rate limit) are
 *  swallowed so one broken plugin doesn't sink the whole enrich
 *  attempt. Returns a flat list of candidates — multiple per plugin
 *  is fine, the caller dedupes by `(plugin, externalId)`. */
export async function lookupAcrossPlugins(
  plugins: LinkPluginAdapter[],
  query:   LinkQuery,
  opts:    { signal?: AbortSignal } = {},
): Promise<LinkCandidate[]> {
  const q = (query.titleNative?.trim() || query.title.trim())
  if (!q) return []

  const results = await Promise.allSettled(
    plugins.map((p) =>
      p.search({ title: query.title, titleNative: query.titleNative }, opts.signal)
        .catch(() => [] as LinkCandidate[]),
    ),
  )

  const out: LinkCandidate[] = []
  for (const r of results) {
    if (r.status === 'fulfilled') out.push(...r.value)
  }
  return out
}


// ── Shared adapter helpers ────────────────────────────────────


/** Proxied fetch helper — every link adapter routes through here so
 *  CORS / referer rules stay consistent. The Discord Activity iframe
 *  needs the proxy path (`/cdn/c/…`); outside the DA, `proxify`
 *  collapses to the upstream URL untouched. */
export function plinkFetch(
  url:    string,
  init:   RequestInit = {},
  signal: AbortSignal | undefined,
): Promise<Response> {
  return fetch(proxify(url), { ...init, signal })
}


/** Coerce arbitrary JSON values to a clean `string | null`. Trims
 *  and drops empty strings. Used by every adapter when mapping
 *  upstream fields onto `LinkCandidate`. */
export function toStr(v: unknown): string | null {
  if (v == null) return null
  if (typeof v === 'string') return v.trim() || null
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  return null
}


/** Flatten a mix of single values / arrays / nulls into a clean
 *  `string[]`. Order-preserving, no de-dup (caller's choice). */
export function toStrArray(...vals: unknown[]): string[] {
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
