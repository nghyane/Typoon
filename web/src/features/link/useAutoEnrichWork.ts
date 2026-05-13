// useAutoEnrichWork — silent, fire-and-forget cross-reference enrichment.
//
// When the user opens a Work whose materials don't yet carry the
// big-namespace IDs (Anilist, MAL, …), this hook fans search out
// across every installed link plugin, normalizes the results, and
// POSTs whatever it found back to the server's
// `/material/{id}/enrich-refs` endpoint. The server's linker then
// re-runs cross_refs matching on the next material import.
//
// Design notes:
//
//   • Silent. No UI surface for the user, no toast, no panel. The
//     value shows up later as a community vote suggestion driven by
//     server-side cross_refs match, OR as an automatic merge once
//     two sibling Works share an ID — neither is interactive on
//     this hook's frontier.
//
//   • One-shot per (work, week). A localStorage flag prevents
//     re-running on every navigation; we re-try after 7 days so a
//     plugin that came online recently gets a chance.
//
//   • Best-effort. Failures (rate limit, network, plugin returns
//     nothing) are swallowed. The hook never throws or blocks
//     render.
//
//   • Skips when cross_refs already populated. The linker only
//     needs ONE plugin hit per Work to do its job; piling more on
//     doesn't help.

import { useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'

import { api, type ApiMaterial, type ApiWorkDetail } from '@shared/api/api'

import { bundledLinkPlugins } from './plugins'
import { lookupAcrossPlugins, type LinkCandidate } from './runtime'


const COOLDOWN_MS = 7 * 24 * 60 * 60 * 1000   // 7 days
const STORAGE_PREFIX = 'enrich:'


export function useAutoEnrichWork(work: ApiWorkDetail | null): void {
  const enrich = useMutation({
    mutationFn: (input: { materialId: number; refs: Record<string, string>; signals: LinkCandidate[] }) =>
      api.enrichMaterialRefs(input.materialId, {
        cross_refs:     input.refs,
        source_signals: input.signals.map((c) => ({
          plugin:        c.plugin,
          confidence:    confidenceFor(c),
          matched_title: c.title,
        })),
      }),
  })

  // Track whether we've already kicked off enrichment for this Work
  // in the current mount. React Strict Mode double-mounts effects;
  // without a ref guard we'd fire twice.
  const firedRef = useRef<number | null>(null)

  useEffect(() => {
    if (!work) return
    if (firedRef.current === work.work.id) return

    // Skip: cross_refs already populated by manifest or a previous
    // enrich call. One plugin hit is enough for the linker.
    if (hasUsefulCrossRefs(work.work.cross_refs)) {
      firedRef.current = work.work.id
      return
    }
    // Skip: cooldown — we tried recently and either found nothing
    // or already submitted.
    const cdKey = STORAGE_PREFIX + work.work.id
    const last = safeReadCooldown(cdKey)
    if (last !== null && Date.now() - last < COOLDOWN_MS) {
      firedRef.current = work.work.id
      return
    }

    // Choose the material whose titles we'll search with. Prefer one
    // carrying `title_native` (kanji is the strongest signal across
    // services); fall back to whichever material the work opens with.
    const primary = pickPrimary(work.work.id, work.materials)
    if (!primary) {
      firedRef.current = work.work.id
      return
    }

    firedRef.current = work.work.id
    // Mark the cooldown NOW so two near-simultaneous mounts (e.g.
    // navigating away and back inside one second) don't fan out
    // duplicate searches. We'll overwrite this on success below.
    safeWriteCooldown(cdKey)

    const ctrl = new AbortController()
    void (async () => {
      try {
        const candidates = await lookupAcrossPlugins(
          bundledLinkPlugins,
          { title: primary.title, titleNative: primary.title_native ?? null },
          { signal: ctrl.signal },
        )
        if (ctrl.signal.aborted) return
        const top = pickTopPerPlugin(candidates)
        const refs = buildRefs(top)
        if (Object.keys(refs).length === 0) return
        enrich.mutate({
          materialId: primary.id,
          refs,
          signals:    top,
        })
      } catch {
        // Best-effort. Swallow errors so a broken plugin or
        // momentary network blip doesn't crash the work page.
      }
    })()

    return () => ctrl.abort()
  // eslint-disable-next-line react-hooks/exhaustive-deps -- mutation is stable
  }, [work?.work.id])
}


// ── Helpers ────────────────────────────────────────────────────


/** True when the Work's `cross_refs` already carries at least one of
 *  the big-namespace IDs the plugins would discover. Empty objects
 *  count as missing; we don't quibble about which specific service
 *  is present. */
function hasUsefulCrossRefs(refs: Record<string, unknown> | null): boolean {
  if (!refs) return false
  return Object.keys(refs).length > 0
}


function pickPrimary(
  _workId:   number,
  materials: ApiMaterial[],
): ApiMaterial | null {
  if (materials.length === 0) return null
  const withNative = materials.find((m) => (m.title_native ?? '').trim().length > 0)
  return withNative ?? materials[0]!
}


/** Keep only the top-ranked candidate per plugin. Anilist returns up
 *  to 3 results per query; we trust the first one most because
 *  Anilist sorts by their internal relevance score. */
function pickTopPerPlugin(candidates: LinkCandidate[]): LinkCandidate[] {
  const byPlugin = new Map<string, LinkCandidate>()
  for (const c of candidates) {
    if (!byPlugin.has(c.plugin)) byPlugin.set(c.plugin, c)
  }
  return [...byPlugin.values()]
}


/** Build the `{namespace: externalId}` payload the enrich endpoint
 *  expects. Drops candidates without an id (already filtered in
 *  runtime, but defensive). */
function buildRefs(candidates: LinkCandidate[]): Record<string, string> {
  const refs: Record<string, string> = {}
  for (const c of candidates) {
    if (c.externalId) refs[c.namespace] = c.externalId
  }
  return refs
}


/** Confidence is a coarse score used for the audit log; not directly
 *  used by the server today, but stamped so a future moderation UI
 *  can see "which plugin claimed this with what confidence". */
function confidenceFor(c: LinkCandidate): number {
  return c.titleNative ? 0.9 : 0.7
}


function safeReadCooldown(key: string): number | null {
  try {
    const v = localStorage.getItem(key)
    if (!v) return null
    const n = Number(v)
    return Number.isFinite(n) ? n : null
  } catch {
    return null
  }
}


function safeWriteCooldown(key: string): void {
  try {
    localStorage.setItem(key, String(Date.now()))
  } catch {
    // localStorage can throw in private mode / quota — fine, the
    // hook degrades to "re-run on every navigation". Not ideal but
    // not harmful.
  }
}
