// Pure resolvers — version pick + chapter navigation + next-plan.
// Tách khỏi hook để test được độc lập, no React, no IO.

import type { HubChapter, HubVersion } from '@features/title/mergeChapters'
import type { SourcePreference } from './store'


/** BCP-47 → primary subtag, lowercase. Accept "vi-VN" / "VI" / null
 *  uniformly. Returns '' for null/empty so callers can branch on
 *  truthiness without `?.` ladders. */
export function normalizeLang(s: string | null | undefined): string {
  if (!s) return ''
  return s.toLowerCase().split(/[-_]/)[0] ?? ''
}


/** Summarise a chapter into a single user-facing state for the
 *  chapter list dropdown. The chapter list intentionally hides
 *  per-version detail (use the source picker for that); each row
 *  just needs to answer "if I tap this, what will I get?".
 *
 *  Honours an explicit source pref so a user reading "AI VI từ EN"
 *  sees that exact draft's status on each row (in-flight → spinner,
 *  done → readable), not the default-fallback policy. */
export type ChapterStateSummary = {
  state: 'done' | 'running' | 'pending' | 'error' | 'raw-only' | 'none'
  label: string
}


export function summarizeChapter(
  ch:         HubChapter,
  targetLang: string | null,
  pref:       SourcePreference | null,
): ChapterStateSummary {
  const lang = normalizeLang(targetLang)

  // Priority 1: explicit pref. If pref points at AI translation from
  // a specific source_lang, surface that draft's state — running /
  // pending / error / done — even if other versions exist.
  if (pref && lang) {
    const v = pickByPref(ch, pref)
    if (v) {
      if (v.kind === 'translation') {
        if (v.state === 'done')                              return { state: 'done',    label: '' }
        if (v.state === 'running' || v.state === 'pending')  return { state: 'running', label: '…' }
        if (v.state === 'error')                             return { state: 'error',   label: '!' }
        return { state: 'pending', label: '…' }
      }
      // raw at target lang — default state, nothing to flag.
      return { state: 'done', label: '' }
    }
  }

  // Priority 2: anything readable in the target lang.
  if (lang) {
    const tx = ch.versions.find(
      (v) => v.kind === 'translation' && v.lang === lang && v.state === 'done',
    )
    if (tx) return { state: 'done', label: '' }
    const rawTgt = ch.versions.find(
      (v) => v.kind === 'raw' && v.lang === lang && !!v.upstreamUrl,
    )
    if (rawTgt) return { state: 'done', label: '' }
    const inflight = ch.versions.find(
      (v) => v.kind === 'translation' && v.lang === lang
          && (v.state === 'running' || v.state === 'pending'),
    )
    if (inflight) return { state: 'running', label: '…' }
    const errored = ch.versions.find(
      (v) => v.kind === 'translation' && v.lang === lang && v.state === 'error',
    )
    if (errored) return { state: 'error', label: '!' }
  }

  // Priority 3: any spawnable raw (user can translate from here).
  // Surface the lang so user sees "ah, có EN thôi" before tapping.
  const anyRaw = ch.versions.find(
    (v) => v.kind === 'raw' && !!v.upstreamUrl && !!v.sourceId,
  )
  if (anyRaw) {
    return { state: 'raw-only', label: anyRaw.lang.toUpperCase() }
  }

  return { state: 'none', label: '—' }
}


/** Pick the version to actually render for a chapter at the user's
 *  preferred target lang. Priority:
 *    1. translation `done` in target lang
 *    2. raw whose lang matches target (read source verbatim)
 *    3. any spawnable raw
 *    4. anything (last resort).
 *  Returns null only when the chapter has no version at all. */
export function pickReadable(
  ch:         HubChapter,
  targetLang: string | null,
): HubVersion | null {
  const lang = normalizeLang(targetLang)

  if (lang) {
    const tx = ch.versions.find(
      (v) => v.kind === 'translation' && v.lang === lang && v.state === 'done',
    )
    if (tx) return tx
  }

  if (lang) {
    const rawTgt = ch.versions.find(
      (v) => v.kind === 'raw' && v.lang === lang && !!v.upstreamUrl,
    )
    if (rawTgt) return rawTgt
  }

  const rawAny = ch.versions.find(
    (v) => v.kind === 'raw' && !!v.upstreamUrl,
  )
  if (rawAny) return rawAny

  return ch.versions[0] ?? null
}


/** Match the user's saved source preference against a chapter's
 *  versions. Returns the matching version, or null when no version
 *  on this chapter fits the pref (caller falls back to
 *  `pickReadable`).
 *
 *  Match keying:
 *    - kind == pref.kind
 *    - lang == pref.lang
 *    - kind == 'translation' \u2192 also require sourceLang match WHEN
 *      pref.sourceLang is set, AND state must be a renderable one
 *      (done / running / pending / error \u2014 reader shows progress).
 *    - kind == 'raw'         \u2192 require installed source plugin so
 *      the chapter is openable.
 *
 *  Among multiple matches, prefer the freshest done translation;
 *  for raws, the first match wins (manifest order). */
export function pickByPref(
  ch:   HubChapter,
  pref: SourcePreference | null,
): HubVersion | null {
  if (!pref) return null
  const lang = normalizeLang(pref.lang)
  if (!lang) return null
  const wantSrc = normalizeLang(pref.sourceLang ?? null)

  if (pref.kind === 'translation') {
    const cands = ch.versions.filter((v) => {
      if (v.kind !== 'translation') return false
      if (v.lang !== lang) return false
      if (wantSrc && normalizeLang(v.sourceLang) !== wantSrc) return false
      return true
    })
    if (cands.length === 0) return null
    // Prefer done; among done prefer newest date; else newest of any.
    const done = cands.filter((v) => v.state === 'done')
    const pool = done.length > 0 ? done : cands
    return pool.reduce((best, v) =>
      !best ? v
      : (v.date ?? '') > (best.date ?? '') ? v
      : best,
    null as HubVersion | null)
  }

  // raw
  return ch.versions.find(
    (v) => v.kind === 'raw' && v.lang === lang
        && !!v.upstreamUrl && !!v.sourceId,
  ) ?? null
}


/** Compute the effective pick for a chapter: saved pref beats
 *  default fallback. Pure so the reader hook can memo it.
 *
 *  Preference:  the user's sticky choice for the work, written the
 *               moment they tap \u0110\u1ecdc in the source picker.
 *  Fallback:    `pickReadable`'s built-in priority \u2014 used the
 *               first time the user opens the work, or when the
 *               current chapter has no version matching the pref. */
export function resolvePicked(
  ch:         HubChapter,
  targetLang: string | null,
  pref:       SourcePreference | null,
): HubVersion | null {
  const byPref = pickByPref(ch, pref)
  if (byPref) return byPref
  return pickReadable(ch, targetLang)
}


export interface NavTarget {
  workId:     number
  numberNorm: string
}


/** Prev/next chapter by INDEX in the merged spine, NOT chapter
 *  number. Filler / extras stay as positions in the spine; the
 *  reader's `empty` status surfaces them rather than silently
 *  skipping. Spine is sorted DESC (latest first) so idx-1 = newer. */
export function resolveNav(
  chapters: HubChapter[],
  current:  HubChapter,
  workId:   number,
): { prev: NavTarget | null; next: NavTarget | null } {
  const idx = chapters.findIndex((c) => c.number === current.number)
  if (idx < 0) return { prev: null, next: null }
  const nextIdx = idx - 1
  const prevIdx = idx + 1
  return {
    next: nextIdx >= 0
      ? { workId, numberNorm: chapters[nextIdx]!.number }
      : null,
    prev: prevIdx < chapters.length
      ? { workId, numberNorm: chapters[prevIdx]!.number }
      : null,
  }
}


export type NextChapterPlan =
  | { kind: 'open-translation';  chapter: HubChapter; version: HubVersion }
  | { kind: 'open-raw';          chapter: HubChapter; version: HubVersion }
  | { kind: 'spawn-from-source'; chapter: HubChapter; raw: HubVersion;
      sourceLang: string }
  | { kind: 'spawn-from-any';    chapter: HubChapter; raw: HubVersion }
  | { kind: 'empty';             chapter: HubChapter }
  | { kind: 'end-of-spine' }


/** 7-step fallback that decides what the end-of-chapter CTA should
 *  do. Used by both EndOfChapterCard and the empty-status page so
 *  the affordance is identical wherever the user lands. */
export function resolveNextPlan(
  chapters:          HubChapter[],
  currentNum:        string,
  targetLang:        string | null,
  readingSourceLang: string | null,
): NextChapterPlan {
  const idx = chapters.findIndex((c) => c.number === currentNum)
  if (idx < 0) return { kind: 'end-of-spine' }
  const nextIdx = idx - 1
  if (nextIdx < 0) return { kind: 'end-of-spine' }
  const next = chapters[nextIdx]!
  const tgt = normalizeLang(targetLang)
  const src = normalizeLang(readingSourceLang)

  if (tgt && src) {
    const v = next.versions.find(
      (x) => x.kind === 'translation' && x.lang === tgt
          && x.state === 'done' && normalizeLang(x.sourceLang) === src,
    )
    if (v) return { kind: 'open-translation', chapter: next, version: v }
  }
  if (tgt) {
    const v = next.versions.find(
      (x) => x.kind === 'translation' && x.lang === tgt && x.state === 'done',
    )
    if (v) return { kind: 'open-translation', chapter: next, version: v }
  }
  if (tgt) {
    const v = next.versions.find(
      (x) => x.kind === 'raw' && x.lang === tgt && !!x.upstreamUrl && !!x.sourceId,
    )
    if (v) return { kind: 'open-raw', chapter: next, version: v }
  }
  if (tgt) {
    const v = next.versions.find(
      (x) => x.kind === 'translation' && x.lang === tgt
          && (x.state === 'pending' || x.state === 'running' || x.state === 'error'),
    )
    if (v) return { kind: 'open-translation', chapter: next, version: v }
  }
  if (src) {
    const raw = next.versions.find(
      (x) => x.kind === 'raw' && normalizeLang(x.lang) === src
          && !!x.upstreamUrl && !!x.sourceId,
    )
    if (raw) return { kind: 'spawn-from-source', chapter: next, raw, sourceLang: src }
  }
  const anyRaw = next.versions.find(
    (x) => x.kind === 'raw' && !!x.upstreamUrl && !!x.sourceId,
  )
  if (anyRaw) return { kind: 'spawn-from-any', chapter: next, raw: anyRaw }

  return { kind: 'empty', chapter: next }
}
