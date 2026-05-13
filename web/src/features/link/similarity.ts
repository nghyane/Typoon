// Title similarity — fuzzy match utilities for the auto-enrich flow.
//
// The naive approach ("first search result wins") burns through trust
// fast: search "Trang" → top hit is some Korean manhwa unrelated to
// the Vietnamese material the user has open. We use a multi-pass
// scoring function to filter that noise before committing any
// cross_refs / metadata to the server.
//
// Three signals, take the max:
//
//   1. Normalized exact match    — short-circuits at 1.0
//   2. Levenshtein ratio         — catches typos, diacritic mismatch
//   3. Token Jaccard             — robust against "Part 2" / "Vol 3"
//                                  and word reorderings
//   4. Token containment         — "honzuki" ⊂ "honzuki no gekokujou"
//                                  hits 0.9 even when ratio is low
//
// Then `isSuspicious` rejects candidates that match weakly AND look
// like spinoffs / volume splits (very common Anilist noise).


/** Strip casing, diacritics, punctuation; collapse whitespace. The
 *  goal is `"Mọt sách lật đổ!" ≡ "mot sach lat do"` so a query in
 *  one orthography matches a candidate in another. */
export function normalizeTitle(s: string): string {
  return s
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')          // strip combining marks
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')        // drop punctuation
    .replace(/\s+/g, ' ')
    .trim()
}


/** Levenshtein-based similarity in 0..1. Uses the two-row dynamic
 *  programming trick — O(n*m) time, O(min(n,m)) space — which is
 *  fine for title-length strings. */
function levenshteinRatio(a: string, b: string): number {
  if (a === b) return 1
  if (!a.length || !b.length) return 0
  // Always iterate over the shorter string to keep the row small.
  if (a.length < b.length) [a, b] = [b, a]
  const prev = new Array(b.length + 1)
  const cur  = new Array(b.length + 1)
  for (let j = 0; j <= b.length; j++) prev[j] = j
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + cost,
      )
    }
    for (let j = 0; j <= b.length; j++) prev[j] = cur[j]
  }
  const dist = prev[b.length]
  return 1 - dist / Math.max(a.length, b.length)
}


/** Word-set Jaccard. `"one piece"` vs `"one piece episode a"` =
 *  0.5 — same two tokens shared, four union. Stable against
 *  trailing volume / part qualifiers. */
function tokenJaccard(a: string, b: string): number {
  const ta = new Set(a.split(' ').filter(Boolean))
  const tb = new Set(b.split(' ').filter(Boolean))
  if (ta.size === 0 || tb.size === 0) return 0
  let inter = 0
  for (const t of ta) if (tb.has(t)) inter++
  return inter / (ta.size + tb.size - inter)
}


/** True when one token set is fully contained in the other. Catches
 *  `"honzuki" ⊂ "honzuki no gekokujou"` where Levenshtein scores
 *  poorly due to length disparity. */
function containmentScore(a: string, b: string): number {
  const ta = new Set(a.split(' ').filter(Boolean))
  const tb = new Set(b.split(' ').filter(Boolean))
  if (ta.size === 0 || tb.size === 0) return 0
  const aInB = [...ta].every((t) => tb.has(t))
  const bInA = [...tb].every((t) => ta.has(t))
  return (aInB || bInA) ? 0.9 : 0
}


/** Take the best of all signals. Returns 0..1; exact normalized
 *  match short-circuits at 1.0. */
export function similarity(a: string, b: string): number {
  const na = normalizeTitle(a)
  const nb = normalizeTitle(b)
  if (!na || !nb) return 0
  if (na === nb) return 1
  return Math.max(
    levenshteinRatio(na, nb),
    tokenJaccard(na, nb),
    containmentScore(na, nb),
  )
}


/** Best similarity of `query` against any of the candidate's
 *  titles (primary + native + synonyms). */
export function bestSimilarity(query: string, titles: (string | null | undefined)[]): number {
  let best = 0
  for (const t of titles) {
    if (!t) continue
    const s = similarity(query, t)
    if (s > best) best = s
    if (best === 1) break
  }
  return best
}


/** Spinoff / volume / part patterns that often top Anilist's search
 *  when a user queries the base series title. Common Anilist noise. */
const SPINOFF_RE = /\b(part|volume|chapter|ch|vol|ep|episode|spinoff|gaiden|side\s*story|extras?)\s*\d?\b/i


/** Reject candidates that match weakly AND smell like a spinoff or
 *  a chapter / volume split. Anilist's relevance scoring is generally
 *  good — when a search misses, the rest of the top-3 are almost
 *  always Part-N / Episode-A clutter. */
export function isSuspiciousCandidate(
  query: string,
  candidateTitles: (string | null | undefined)[],
  score:  number,
): boolean {
  // Exact / very high match — trust it regardless of "Part" in title.
  if (score >= 0.92) return false

  // Score < 0.85 and the candidate's display titles look like a
  // volume / spinoff → bail.
  if (score < 0.85) {
    const joined = candidateTitles.filter(Boolean).join(' ')
    if (SPINOFF_RE.test(joined)) return true
  }

  // Very short query against a much longer candidate, with no
  // containment win. "Trang" → "Trang Naked Apron" should NOT
  // pass on a 4-char query.
  const queryLen = normalizeTitle(query).length
  if (queryLen > 0 && queryLen < 4 && score < 0.95) return true

  // Length ratio: a 4-char query matching a 30-char title with
  // score 0.7 is suspicious — probably a coincidental fragment.
  const longest = Math.max(
    queryLen,
    ...candidateTitles.map((t) => normalizeTitle(t ?? '').length),
  )
  const ratio = longest > 0 ? queryLen / longest : 0
  if (ratio > 0 && ratio < 0.35 && score < 0.9) return true

  return false
}


/** Confidence tiers the enrich pipeline decides on. Drives whether
 *  metadata commits in full, partial (cross_refs only), or skips. */
export type MatchDecision = 'accept' | 'maybe' | 'skip'

export function decideMatch(score: number): MatchDecision {
  if (score >= 0.85) return 'accept'
  if (score >= 0.65) return 'maybe'
  return 'skip'
}
