// useAutoEnrichWork — silent metadata enrichment.
//
// Flow on Work-page mount:
//
//   1. Skip when the primary material already has cross_refs +
//      title_native + a target_lang entry in title_locale (the
//      enrich would just rewrite what's already there).
//   2. Honor a 7-day cooldown per Work (localStorage). Cooldown is
//      only set AFTER a successful commit (or a clean "no matches"
//      result) — transient lookup / network failures retry next
//      visit instead of locking the Work out for a week.
//   3. Fan search out across `bundledLinkPlugins` (MangaBaka +
//      MangaDex today).
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
// React Strict Mode note: the effect's cleanup aborts the in-flight
// fetch on unmount. Strict Mode mount→cleanup→remount calls the
// effect body twice — the second run starts a fresh AbortController
// and a fresh fetch, so the work always completes. No ref guard is
// needed; the useEffect deps (`work?.work.id`) already collapse
// re-renders that don't change the work id.
//
// No UI surface — the enriched data shows up as title language
// improvements + auto-merge candidates in the LinkSuggestionPanel
// once the linker round-trips it.

import { useEffect } from 'react'
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
// Bump the prefix whenever the enrich pipeline gains new data that
// older cooldown stamps would gate out. Old keys never expire on
// their own (localStorage doesn't gc), so the suffix is the only way
// to force a one-time re-run across the whole user base.
//
// v2 (2026-05): plugins now emit per-lang `title_locale` (was
// hardcoded `en`/`ja` in the merge layer). Old enrich rows only have
// the two-locale subset; bumping the prefix lets every Work re-run
// once so resolver finds e.g. `vi` titles when the plugin actually
// has them.
const STORAGE_PREFIX = 'enrich:v2:'


interface EnrichPayload {
  cross_refs?:   Record<string, string>
  title_native?: string
  title_alt?:    string[]
  title_locale?: Record<string, string>
  start_year?:   number
  cover_url?:    string
  source_signals: Array<{
    plugin:        string
    confidence:    number
    matched_title: string | null
  }>
}


export function useAutoEnrichWork(
  work:          ApiWorkDetail | null,
  /** Viewer's reading lang. When provided, an otherwise-enriched
   *  material whose `title_locale` is missing this lang triggers a
   *  re-enrich. Lets a plugin upgrade (e.g. mangabaka started
   *  surfacing per-lang locales) backfill old rows lazily without a
   *  bulk migration — every Work the viewer opens self-heals on
   *  first hit. */
  viewerLang?:   string | null,
): void {
  const enrich = useMutation({
    mutationFn: (input: { materialId: number; payload: EnrichPayload }) =>
      api.enrichMaterialMetadata(input.materialId, input.payload),
  })

  useEffect(() => {
    if (!work) return

    const primary = pickPrimary(work.materials)
    if (!primary) return
    if (isAlreadyEnriched(primary, viewerLang ?? null)) return

    // Cooldown: 7-day window keyed by Work id + pipeline version.
    // Bumping the `STORAGE_PREFIX` suffix forces every Work to
    // re-run once after a pipeline upgrade (see the v2 note). The
    // commit() path stamps the key on every successful run — `ok` or
    // `nothing matched` — so the cooldown still gates the "nobody
    // can identify this manga" case from hammering the network.
    const cdKey = STORAGE_PREFIX + work.work.id
    if (withinCooldown(cdKey)) return

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
        if (!hasContent(payload)) {
          // Nothing to commit — but the lookup itself succeeded, so
          // cooldown applies: re-running tomorrow will just hit the
          // same dry plugins. The 7-day window is the right cadence
          // to let candidate plugins surface new matches.
          writeCooldown(cdKey)
          return
        }

        await enrich.mutateAsync({ materialId: primary.id, payload })
        // Only stamp cooldown AFTER a successful commit. A transient
        // network / plugin failure should re-try on the next visit
        // instead of locking the Work out for a week.
        writeCooldown(cdKey)
      } catch {
        // Best-effort. Plugin / network / mutation errors degrade
        // silently — cooldown stays unset so the next page mount
        // gets a clean retry.
      }
    })()

    return () => ctrl.abort()
  // eslint-disable-next-line react-hooks/exhaustive-deps -- mutation is stable
  }, [work?.work.id])
}


// ── Pure helpers ───────────────────────────────────────────────


/** Pick the material whose titles we'll search with. The auto-enrich
 *  fanout queries upstream metadata services (MangaBaka, MangaDex,
 *  …) that index by NATIVE / romanized titles — VN scanlator titles
 *  ("Cầm Dao Mổ Heo Chém Bay Vạn Giới") rarely hit, so picking the
 *  VN ext-upload material as primary on a multi-source Work would
 *  cause the entire enrich to come back empty.
 *
 *  Priority:
 *    1. Material with a non-empty `title_native`        (strongest).
 *    2. Material whose `title` contains CJK / Hangul    (HappyMH-zh,
 *       MangaDex with CJK display title — upstream native, no need
 *       for the title_native field to be populated).
 *    3. Source-backed material (`source != null`)       — upstream
 *       romanized title is still a stronger signal than a scanlator
 *       VN title.
 *    4. `materials[0]`                                  — last resort.
 */
function pickPrimary(materials: ApiMaterial[]): ApiMaterial | null {
  if (materials.length === 0) return null

  const withNative = materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  )
  if (withNative) return withNative

  const withCjkTitle = materials.find(
    (m) => containsCjk(m.title),
  )
  if (withCjkTitle) return withCjkTitle

  const sourceBacked = materials.find((m) => m.source != null)
  if (sourceBacked) return sourceBacked

  return materials[0]!
}


/** True when the string carries CJK Unified Ideographs, Hiragana,
 *  Katakana, or Hangul. These ranges signal "native script title"
 *  cleanly enough for the enrich primary picker — we don't need to
 *  be exhaustive (Cyrillic / Thai etc. are not enrichment targets
 *  for the bundled plugins). */
function containsCjk(s: string | null | undefined): boolean {
  if (!s) return false
  return /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]/.test(s)
}


/** Skip the round-trip when the material already carries the data
 *  an enrich would have written. Defines "enough":
 *    • cross_refs has at least one known namespace, AND
 *    • title_native is filled, AND
 *    • title_locale has at least two languages, AND
 *    • if `viewerLang` is given, `title_locale[viewerLang]` is
 *      present (so a viewer reading in 'vi' triggers a re-enrich
 *      even when the row already has en+ja from a previous run).
 *
 *  Anything less, we try again; the merge is additive so re-running
 *  on a partially-enriched row only adds, never overwrites. */
function isAlreadyEnriched(
  m:          ApiMaterial,
  viewerLang: string | null,
): boolean {
  const hasIdRef = !!m.cross_refs && (
    'anilist' in m.cross_refs
    || 'mdex_uuid' in m.cross_refs
    || 'mal' in m.cross_refs
  )
  const hasNative = (m.title_native ?? '').trim().length > 0
  const locale = m.title_locale ?? {}
  const localeCount = Object.values(locale).filter(
    (v) => v && v.trim(),
  ).length
  if (!hasIdRef || !hasNative || localeCount < 2) return false
  const target = (viewerLang ?? '').trim().toLowerCase().split(/[-_]/)[0]
  if (target && !(locale[target] && locale[target].trim())) return false
  return true
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
  let cover:       string | null = null

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

    // Merge lang-tagged display titles. Plugin order determines
    // priority: a key that's already populated by an earlier plugin
    // is kept (first-writer-wins) so the registry order in
    // `bundledLinkPlugins` is also the priority order.
    //
    // Plus the legacy `titleEnglish` / `titleNative` flags fill in
    // `en` / a CJK key when the plugin didn't surface them via
    // `titleLocale` — back-compat for adapters not yet emitting the
    // full locale map.
    for (const [lang, value] of Object.entries(top.c.titleLocale)) {
      if (!value || titleLocale[lang]) continue
      titleLocale[lang] = value
    }
    if (top.c.titleEnglish && !titleLocale['en']) {
      titleLocale['en'] = top.c.titleEnglish
    }
    // Heuristic backstop for `titleNative`: only attach to a CJK key
    // when we can detect the script. Hardcoding `'ja'` (the old
    // behaviour) labelled Korean / Chinese natives as Japanese in
    // the resolver and the title_locale map became misleading.
    if (top.c.titleNative) {
      const cjk = detectCjkScript(top.c.titleNative)
      if (cjk && !titleLocale[cjk]) titleLocale[cjk] = top.c.titleNative
    }

    for (const s of top.c.synonyms) titleAltSet.add(s)
    if (top.c.startYear && !startYear) startYear = top.c.startYear
    // Cover is first-writer-wins on the storage side; we still pick
    // the FIRST plugin's accepted cover so ordering (MangaBaka over
    // MangaDex) drives the choice. Per-material existing covers are
    // preserved by the COALESCE on `cover_url`.
    if (!cover && top.c.cover) cover = top.c.cover
  }

  const payload: EnrichPayload = { source_signals: signals }
  if (Object.keys(crossRefs).length > 0)   payload.cross_refs   = crossRefs
  if (titleNative)                         payload.title_native = titleNative
  if (titleAltSet.size > 0)                payload.title_alt    = [...titleAltSet]
  if (Object.keys(titleLocale).length > 0) payload.title_locale = titleLocale
  if (startYear)                           payload.start_year   = startYear
  if (cover)                               payload.cover_url    = cover
  return payload
}


function hasContent(p: EnrichPayload): boolean {
  return (
    !!p.cross_refs
    || !!p.title_native
    || !!p.title_alt
    || !!p.title_locale
    || p.start_year != null
    || !!p.cover_url
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


/** Detect the dominant CJK script of a string and return a BCP-47
 *  primary subtag. Used by the enrich pipeline to label a plugin's
 *  `titleNative` with the right `title_locale` key when the plugin
 *  didn't already split natives per-lang.
 *
 *  We classify by the FIRST CJK code point we see. That's enough
 *  for the manga case: a Japanese title is overwhelmingly hiragana/
 *  katakana, a Korean title is hangul, a Chinese title is purely
 *  Han ideographs. Returns null for non-CJK strings (Latin, Cyrillic,
 *  Thai...) — the caller leaves them alone rather than guessing. */
function detectCjkScript(s: string): 'ja' | 'ko' | 'zh' | null {
  for (const ch of s) {
    const cp = ch.codePointAt(0)
    if (cp == null) continue
    // Hiragana 3040–309F | Katakana 30A0–30FF | Katakana ext 31F0–31FF
    if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF)) {
      return 'ja'
    }
    // Hangul Syllables AC00–D7AF | Jamo 1100–11FF | Compat A 3130–318F
    if ((cp >= 0xAC00 && cp <= 0xD7AF) ||
        (cp >= 0x1100 && cp <= 0x11FF) ||
        (cp >= 0x3130 && cp <= 0x318F)) {
      return 'ko'
    }
    // CJK Unified 4E00–9FFF — claim zh as the safest default when
    // the string is purely Han with no hiragana/katakana/hangul.
    if (cp >= 0x4E00 && cp <= 0x9FFF) {
      return 'zh'
    }
  }
  return null
}
