// Work title / cover / primary-material resolvers.
//
// One Work groups N sibling materials backed by different sources.
// Hub / community feed / recent reads all need ONE canonical display
// (title + cover + handle) consistent with the viewer's preferred
// reading language. Same chain runs server-side in
// `typoon/storage/postgres.py` so SSR-style endpoints (community feed,
// recent reads) return the same answer the SPA computes locally.
//
// Title priority (highest first):
//   1. title_locale[targetLang]               (any material)
//   2. material whose languages cover target  → its title
//   3. any material's title_native            (native script)
//   4. materials[0].title                     (last resort)
//
// We deliberately DON'T fall through to a "fallback lang priority"
// chain (en > ja > ko > zh > vi). When the viewer asked for `vi`
// and no material is `vi`, that chain surfaced Romaji ("Honzuki...")
// which is illegible to BOTH the Vietnamese viewer and native
// speakers. Native CJK script is the unambiguous "real name" — show
// it instead, then drop to whatever's available.
//
// `titleNative` (italic subtitle under h1) is the first material's
// title_native regardless of which one supplied the primary title —
// cosmetic, doesn't drive identity.

import type { ApiMaterial } from '@shared/api/api'


export interface ResolvedWorkTitle {
  title:       string
  titleNative: string | null
  /** Material the title came from. Null when the title came from
   *  `title_locale` aggregated across multiple materials. */
  sourceMaterialId: number | null
}


export function resolveWorkTitle(
  materials:  ApiMaterial[],
  targetLang: string | null,
): ResolvedWorkTitle {
  if (materials.length === 0) {
    return { title: '—', titleNative: null, sourceMaterialId: null }
  }
  const native = firstNative(materials)
  const norm   = normalizeLang(targetLang)

  if (norm) {
    for (const m of materials) {
      const t = m.title_locale?.[norm]?.trim()
      if (t) return { title: t, titleNative: native, sourceMaterialId: m.id }
    }
    const byLang = pickByLang(materials, norm)
    if (byLang) {
      return { title: byLang.title, titleNative: native, sourceMaterialId: byLang.id }
    }
  }

  const withNative = materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  )
  if (withNative && (withNative.title_native ?? '').trim()) {
    return {
      title:            withNative.title_native!.trim(),
      titleNative:      native,
      sourceMaterialId: withNative.id,
    }
  }

  const first = materials[0]!
  return { title: first.title, titleNative: native, sourceMaterialId: first.id }
}


export interface ResolvedWorkCover {
  coverUrl:   string | null
  materialId: number | null
}


/** Cover priority — viewer-lang material first, then any cover.
 *  No fallback lang chain: covers are mostly identical across sources
 *  for the same Work, so picking "the en cover before the zh cover"
 *  doesn't help the viewer. Just take whatever's there. */
export function resolveWorkCover(
  materials:  ApiMaterial[],
  targetLang: string | null,
): ResolvedWorkCover {
  if (materials.length === 0) {
    return { coverUrl: null, materialId: null }
  }
  const norm = normalizeLang(targetLang)
  if (norm) {
    const byLang = materials.find(
      (m) => hasLang(m, norm) && !!m.cover_url,
    )
    if (byLang) return { coverUrl: byLang.cover_url, materialId: byLang.id }
  }
  const anyCover = materials.find((m) => !!m.cover_url)
  if (anyCover) return { coverUrl: anyCover.cover_url, materialId: anyCover.id }
  return { coverUrl: null, materialId: null }
}


/** Pick the material that backs the hero metadata strip (status,
 *  bookmark, description). Same chain as `resolveWorkTitle`. */
export function pickPrimaryMaterial(
  materials:  ApiMaterial[],
  targetLang: string | null,
): ApiMaterial | null {
  if (materials.length === 0) return null
  const norm = normalizeLang(targetLang)
  if (norm) {
    const byLocale = materials.find((m) => m.title_locale?.[norm]?.trim())
    if (byLocale) return byLocale
    const byLang = pickByLang(materials, norm)
    if (byLang) return byLang
  }
  const withNative = materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  )
  if (withNative) return withNative
  return materials[0]!
}


function hasLang(m: ApiMaterial, code: string): boolean {
  return (m.languages ?? []).some((l) => normalizeLang(l) === code)
}


function pickByLang(materials: ApiMaterial[], code: string): ApiMaterial | null {
  return materials.find((m) => hasLang(m, code)) ?? null
}


function normalizeLang(lang: string | null | undefined): string | null {
  if (!lang) return null
  const n = lang.toLowerCase().split(/[-_]/)[0]
  return n || null
}


function firstNative(materials: ApiMaterial[]): string | null {
  for (const m of materials) {
    const n = (m.title_native ?? '').trim()
    if (n) return n
  }
  return null
}


/** All known alternate titles for a Work, deduped, with `currentTitle`
 *  removed (it's already shown as the H1). Order: every material's
 *  `title`, then `title_native`, then `title_alt[]`, then
 *  `title_locale{}` values — first occurrence wins for stability.
 *
 *  Used by the WorkHero "▾ alt titles" disclosure. Empty result =
 *  no disclosure rendered. */
export function collectAltTitles(
  materials:    ApiMaterial[],
  currentTitle: string,
): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  const cur = currentTitle.trim()
  if (cur) seen.add(cur)

  const push = (raw: string | null | undefined) => {
    if (!raw) return
    const t = raw.trim()
    if (!t || seen.has(t)) return
    seen.add(t)
    out.push(t)
  }

  for (const m of materials) push(m.title)
  for (const m of materials) push(m.title_native)
  for (const m of materials) for (const a of m.title_alt ?? []) push(a)
  for (const m of materials) {
    const loc = m.title_locale
    if (!loc) continue
    for (const v of Object.values(loc)) push(v)
  }
  return out
}
