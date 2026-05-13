// Work title — resolve the display string for a Work from its
// sibling materials, biased toward the viewer's reading language.
//
// Real use case: a Vietnamese reader opens a manga that the OTruyen
// scanlation team translated and the same series is also indexed on
// HappyMH as raw Japanese. They want to see the title THEY know
// (VI), not a romanized English version they'd have to translate
// back.
//
// Priority (highest first):
//
//   1. The material the user explicitly picked via `?src=`
//      (`activeMaterial.title`). Their choice is final — even if
//      it's a non-target-lang source, respect it.
//
//   2. Any material's `title_locale[targetLang]` — multilingual map
//      populated by the auto-enrich flow (MangaDex altTitles +
//      Anilist title.english). The cleanest cross-source signal we
//      have for translated names.
//
//   3. Material whose `languages` covers the viewer's `target_lang`
//      — direct source title (OTruyen's Vietnamese, MangaDex VI fork,
//      etc).
//
//   4. Material with a `title_native` set — kanji / hangul anchors
//      identity when the previous tiers miss.
//
//   5. `materials[0]` — oldest fallback so something always renders.
//
// The native subtitle (the small italic line under the h1) is the
// first material's `title_native` regardless of which one supplied
// the primary title — purely cosmetic, doesn't drive identity.

import type { ApiMaterial } from '@shared/api/api'


export interface ResolvedWorkTitle {
  title:       string
  titleNative: string | null
  /** Material the title came from. Null when the title came from
   *  `title_locale` aggregated across multiple materials. */
  sourceMaterialId: number | null
}


export function resolveWorkTitle(
  materials:      ApiMaterial[],
  activeMaterial: ApiMaterial | null,
  targetLang:     string | null,
): ResolvedWorkTitle {
  if (materials.length === 0) {
    return { title: '—', titleNative: null, sourceMaterialId: null }
  }

  // 1. User pick wins.
  if (activeMaterial) {
    return {
      title:            activeMaterial.title,
      titleNative:      firstNative(materials),
      sourceMaterialId: activeMaterial.id,
    }
  }

  // 2. Enriched title_locale[targetLang] — any material that has it.
  const norm = normalizeLang(targetLang)
  if (norm) {
    for (const m of materials) {
      const t = m.title_locale?.[norm]?.trim()
      if (t) {
        return {
          title:            t,
          titleNative:      firstNative(materials),
          sourceMaterialId: m.id,
        }
      }
    }
    // 3. Material whose own languages match the target lang.
    const byLang = materials.find((m) =>
      (m.languages ?? []).some((l) => normalizeLang(l) === norm),
    )
    if (byLang) {
      return {
        title:            byLang.title,
        titleNative:      firstNative(materials),
        sourceMaterialId: byLang.id,
      }
    }
  }

  // 4. Material with title_native — romanized title is the most
  //    reliable cross-language anchor when nothing else fits.
  const withNative = materials.find(
    (m) => (m.title_native ?? '').trim().length > 0,
  )
  if (withNative) {
    return {
      title:            withNative.title,
      titleNative:      firstNative(materials),
      sourceMaterialId: withNative.id,
    }
  }

  // 5. Default.
  const first = materials[0]!
  return {
    title:            first.title,
    titleNative:      firstNative(materials),
    sourceMaterialId: first.id,
  }
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
