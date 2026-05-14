// Single source of truth for "create a source-backed material from a
// manifest hit". Every UI flow that imports a material — adding to
// library, manually linking siblings on the Work hub, future quick-add
// extensions — funnels through `importMaterialFromHit`. There is no
// other supported path.
//
// Why centralise:
//
//   The previous shape had `api.importMaterial({...})` called from
//   several callsites, each hand-rolling the `ImportBody`. When a
//   field was added to the schema (`languages`, `title_native`,
//   `title_locale`...), one or two callsites got updated and the
//   others silently dropped the data.
//
//   Concrete bug: `LinkSearchModal` skipped `languages`. Otruyen
//   imports through the modal landed with `languages={}`; the title
//   resolver lost the `vi` signal and surfaced the Mangadex Romaji
//   instead of the Otruyen Vietnamese title.
//
//   The fix is structural: callers only know `(hit, detail?)`. This
//   module owns translation to the wire shape, so adding a field to
//   `ImportBody` only edits ONE function.
//
// Detail fetch is folded in: pass a pre-fetched `MangaDetail` if you
// already have one (URL-paste form does), otherwise this helper
// fetches it. Snapshot from the search hit is the fallback when the
// detail call fails — partial data beats blocking the import on a
// flaky upstream.

import { fetchMangaDetail } from '@features/browse/manifest/runtime'
import type { MangaDetail } from '@features/browse/manifest/types'
import type { SearchHit } from '@features/library/addManga/fanoutSearch'
import { api, type ApiMaterial } from '@shared/api/api'


/** The exact shape `POST /api/material/import` accepts. Mirrors
 *  `ImportBody` in `typoon/api/routes/material.py`. Kept in this
 *  module rather than re-exported from `api.ts` so callers can't
 *  bypass the helper and hand-build the body. */
export interface MaterialImportPayload {
  source:        string
  upstream_ref:  string
  title:         string
  cover_url:     string | null
  description:   string | null
  author:        string | null
  status:        string | null
  languages:     string[]
  title_native:  string | null
  nsfw:          boolean
}


/** Build the wire payload from a search hit + (optional) resolved
 *  detail. Pure function, no I/O — easy to unit-test, easy to reason
 *  about. Detail wins on every field it carries; missing detail
 *  falls back to the hit snapshot, then to the manifest defaults.
 *
 *  `languages` falls back to the manifest's static declaration
 *  (`["vi"]` for Otruyen, `["zh"]` for Happymh) so a source whose
 *  detail page doesn't enumerate available langs still seeds the
 *  resolver with the right signal. */
export function buildMaterialPayload(
  hit:    SearchHit,
  detail: MangaDetail | null,
): MaterialImportPayload {
  const { source, manga } = hit
  const manifest = source.manifest
  return {
    source:       manifest.id,
    upstream_ref: manga.url,
    title:        detail?.title       || manga.title,
    cover_url:    detail?.cover       ?? manga.cover ?? null,
    description:  detail?.description ?? null,
    author:       detail?.author      ?? null,
    status:       detail?.status      ?? null,
    languages:    detail?.availableLanguages ?? manifest.languages,
    title_native: null,    // sources don't surface CJK natives yet
    nsfw:         !!manifest.nsfw,
  }
}


/** Import a material from a search hit. If `detail` is omitted, this
 *  fetches the canonical detail page first; on failure it falls back
 *  to the hit snapshot rather than aborting (a flaky upstream
 *  shouldn't block "thêm vào thư viện"). Returns the persisted
 *  `ApiMaterial` row.
 *
 *  Idempotent on the server — the same `(source, upstream_ref)` pair
 *  returns the existing row; safe to call from both first-import and
 *  manual-link flows without dedupe juggling on the client. */
export async function importMaterialFromHit(
  hit:    SearchHit,
  detail?: MangaDetail | null,
): Promise<ApiMaterial> {
  const resolved = detail !== undefined
    ? detail
    : await fetchMangaDetail(hit.source.manifest, hit.manga.url)
        .catch(() => null)
  return api.importMaterial(buildMaterialPayload(hit, resolved))
}
