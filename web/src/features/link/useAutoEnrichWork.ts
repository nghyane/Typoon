// useAutoEnrichWork — silent metadata enrichment.
//
// Flow on Work-page mount:
//
//   1. Skip when the primary material already has cross_refs +
//      title_native + a target_lang entry in title_locale (the
//      enrich would just rewrite what's already there).
//   2. Honor a 7-day cooldown per Work (localStorage).
//   3. Fan search out across `bundledLinkPlugins` (Anilist today,
//      MangaDex next).
//   4. Score each candidate against the material's titles using the
//      multi-pass similarity in `similarity.ts`; reject obvious
//      noise (spinoffs, suspicious length ratios) via
//      `isSuspiciousCandidate`.
//   5. Decide per-candidate:
//        accept (≥ 0.85)  → commit cross_refs + title_native +
//                            title_locale + synonyms + start_year
//        maybe  (≥ 0.65)  → commit cross_refs ONLY; the linker
//                            verifies via cross-vote later
//        skip   (< 0.65)  → no write, cooldown applies
//   6. POST the merged payload to `/material/{id}/enrich-metadata`.
//
// No UI surface — the enriched data shows up as title language
// improvements + auto-merge candidates in the LinkSuggestionPanel
// once the linker round-trips it.

import { useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api, type ApiMaterial, type ApiWorkDetail } from '@shared/api/api'

import { bundledLinkPlugins } from './plugins'
import {
  lookupAcrossPlugins, type LinkCandidate, type LinkPlugin,
} from './runtime'
import {
  bestSimilarity, decideMatch, isSuspiciousCandidate,
} from './similarity'


const COOLDOWN_MS = 7 * 24 * 60 * 60 * 1000   // 7 days
const STORAGE_PREFIX = 'enrich:'


interface EnrichPayload {
  cross_refs?:   Record<string, string>
  title_native?: string
  title_alt?:    string[]
  title_locale?: Record<string, string>
  start_year?:   number
  source_signals: Array<{
    plugin:        string
    confidence:    number
    matched_title: string | null
  }>
}


export function useAutoEnrichWork(work: ApiWorkDetail | null): void {
  const enrich = useMutation({
    mutationFn: (input: { materialId: number; payload: EnrichPayload }) =>
      api.enrichMaterialMetadata(input.materialId, input.payload),
  })

  // React Strict Mode double-mounts; ref guard prevents firing twice
  // on the same Work in one render cycle.
  const firedRef = useRef<number | null>(null)

  useEffect(() => {
    if (!work) return
    if (firedRef.current === work.work.id) return

    const primary = pickPrimary(work.materials)
    if (!primary) {
      firedRef.current = work.work.id
      return
    }
    if (isAlreadyEnriched(primary)) {
      firedRef.current = work.work.id
      return
    }
    const cdKey = STORAGE_PREFIX + work.work.id
    if (withinCooldown(cdKey)) {
      firedRef.current = work.work.id
      return
    }

    firedRef.current = work.work.id
    writeCooldown(cdKey)

    const ctrl = new AbortController()
    void (async () => {
      try {
        const candidates = await lookupAcrossPlugins(
          bundledLinkPlugins,
          {
            title:       primary.title,
            titleNative: primary.title_native ?? null,
          },
          { signal: ctrl.signal },
        )
        if (ctrl.signal.aborted) return

        const payload = buildPayload(bundledLinkPlugins, primary, candidates)
        if (!hasContent(payload)) return

        enrich.mutate({ materialId: primary.id, payload })
      } catch {
        // Best-effort. Plugin / network errors degrade silently.
      }
    })()

    return () => ctrl.abort()
  // eslint-disable-next-line react-hooks/exhaustive-deps -- mutation is stable
  }, [work?.work.id])
}


// ── Pure helpers ───────────────────────────────────────────────


/** Pick the material whose titles we'll search with. Prefer one
 *  with `title_native` (kanji / hangul = strongest cross-language
 *  anchor), else the first material in the work. */
function pickPrimary(materials: ApiMaterial[]): ApiMaterial | null {
  if (materials.length === 0) return null
  const withNative = materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  )
  return withNative ?? materials[0]!
}


/** Skip the round-trip when the material already carries the data
 *  an enrich would have written. Defines "enough":
 *    • cross_refs has at least one known namespace, AND
 *    • title_native is filled, AND
 *    • title_locale has at least two languages.
 *
 *  Anything less, we try again; the merge is additive so re-running
 *  on a partially-enriched row only adds, never overwrites. */
function isAlreadyEnriched(m: ApiMaterial): boolean {
  const hasIdRef = !!m.cross_refs && (
    'anilist' in m.cross_refs
    || 'mdex_uuid' in m.cross_refs
    || 'mal' in m.cross_refs
  )
  const hasNative = (m.title_native ?? '').trim().length > 0
  const localeCount = m.title_locale
    ? Object.values(m.title_locale).filter((v) => v && v.trim()).length
    : 0
  return hasIdRef && hasNative && localeCount >= 2
}


/** Score every candidate against the material's known titles, drop
 *  the suspicious ones, group by plugin, take the top per plugin,
 *  and translate the survivors into an `EnrichPayload`.
 *
 *  Per-plugin scoring: a plugin's candidates compete only with their
 *  own siblings — we never demote Anilist's best because MangaDex
 *  had a better one. Each plugin contributes the namespace ID it's
 *  most confident about. */
function buildPayload(
  plugins:    LinkPlugin[],
  primary:    ApiMaterial,
  candidates: LinkCandidate[],
): EnrichPayload {
  const queries = [primary.title_native, primary.title].filter(
    (t): t is string => !!t && t.trim().length > 0,
  )

  // Bucket candidates by plugin id so we can pick top-per-plugin.
  const byPlugin = new Map<string, LinkCandidate[]>()
  for (const c of candidates) {
    if (!byPlugin.has(c.plugin)) byPlugin.set(c.plugin, [])
    byPlugin.get(c.plugin)!.push(c)
  }

  const signals: EnrichPayload['source_signals'] = []
  const crossRefs: Record<string, string> = {}
  const titleLocale: Record<string, string> = {}
  const titleAltSet = new Set<string>()
  let titleNative: string | null = null
  let startYear:   number | null = null

  // Plugin order is the registry order — plugins listed first in
  // `bundledLinkPlugins` get priority on `title_locale` conflicts.
  for (const plugin of plugins) {
    const bucket = byPlugin.get(plugin.id)
    if (!bucket || bucket.length === 0) continue

    const scored = bucket.map((c) => {
      const score = bestSimilarity(
        queries[0] ?? '',
        [c.title, c.titleEnglish, c.titleNative, ...c.synonyms],
      )
      return { c, score }
    })

    // Reject suspicious shapes before picking the top.
    const clean = scored.filter(({ c, score }) =>
      !isSuspiciousCandidate(
        queries[0] ?? '',
        [c.title, c.titleEnglish, c.titleNative, ...c.synonyms],
        score,
      ),
    )
    if (clean.length === 0) continue

    clean.sort((a, b) => b.score - a.score)
    const top = clean[0]!
    const decision = decideMatch(top.score)
    if (decision === 'skip') continue

    // Every accepted / maybe candidate contributes its namespace ID.
    if (top.c.externalId) crossRefs[plugin.namespace] = top.c.externalId

    signals.push({
      plugin:        plugin.id,
      confidence:    top.score,
      matched_title: top.c.title,
    })

    if (decision !== 'accept') continue

    // Accept tier — commit display metadata, not just the id.
    if (!titleNative && top.c.titleNative) titleNative = top.c.titleNative
    if (top.c.titleEnglish) {
      titleLocale['en'] = titleLocale['en'] ?? top.c.titleEnglish
    }
    if (top.c.titleNative) {
      titleLocale['ja'] = titleLocale['ja'] ?? top.c.titleNative
    }
    for (const s of top.c.synonyms) titleAltSet.add(s)
    if (top.c.startYear && !startYear) startYear = top.c.startYear
  }

  const payload: EnrichPayload = { source_signals: signals }
  if (Object.keys(crossRefs).length > 0)   payload.cross_refs   = crossRefs
  if (titleNative)                         payload.title_native = titleNative
  if (titleAltSet.size > 0)                payload.title_alt    = [...titleAltSet]
  if (Object.keys(titleLocale).length > 0) payload.title_locale = titleLocale
  if (startYear)                           payload.start_year   = startYear
  return payload
}


function hasContent(p: EnrichPayload): boolean {
  return (
    !!p.cross_refs
    || !!p.title_native
    || !!p.title_alt
    || !!p.title_locale
    || p.start_year != null
  )
}


function withinCooldown(key: string): boolean {
  try {
    const v = localStorage.getItem(key)
    if (!v) return false
    const n = Number(v)
    if (!Number.isFinite(n)) return false
    return Date.now() - n < COOLDOWN_MS
  } catch {
    return false
  }
}


function writeCooldown(key: string): void {
  try {
    localStorage.setItem(key, String(Date.now()))
  } catch {
    // localStorage can throw in private mode / quota — fine, the
    // hook degrades to "re-run on every navigation".
  }
}
