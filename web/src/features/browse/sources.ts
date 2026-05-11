// Sources registry — single source of truth for "what sources are
// installed". Phase 1a ships only bundled manifests from
// `packages/manga-sources/`; phase 1c adds repo + file install.
//
// Bundled manifests are imported at build time via `import.meta.glob`
// so Vite ships them with the bundle (no runtime fetch needed).

import { create } from 'zustand'
import { useShallow } from 'zustand/react/shallow'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { InstalledSource, SourceManifest } from './manifest/types'

// ── bundled manifests ──────────────────────────────────────────────

// `eager: true` inlines every JSON at build time as a default export.
// Path is relative to *this file*. From web/src/features/browse/ we
// need to go up 4 levels to reach the repo root, then into packages/.
//
// NOTE: relative paths don't follow Vite aliases for `glob`, so we
// can't use `@packages/...` here.
const BUNDLED: Record<string, { default: SourceManifest }> =
  import.meta.glob('../../../../packages/manga-sources/*.json', { eager: true })

// `index.json` lives alongside the manifests; filter it out — it is
// the bundled-list manifest, not a source manifest.
function isManifest(mod: { default: unknown }): mod is { default: SourceManifest } {
  const m = mod.default as Partial<SourceManifest> | undefined
  // Accept any manifest with an id + name. External manifests have
  // `endpoints` — internal manifests don't, and that's fine because
  // the runtime branches on `kind === 'internal'` before touching
  // endpoint data.
  return !!m?.id && !!m?.name
}

export const bundledManifests: SourceManifest[] = Object.values(BUNDLED)
  .filter(isManifest)
  .map((m) => m.default)

// ── installed source store ─────────────────────────────────────────

interface SourcesStore {
  /** Installed sources by manifest.id. */
  sources: Record<string, InstalledSource>
  /** Hydrated from bundled list on first run; idempotent across hot reloads. */
  ensureBundled: () => void
  setEnabled: (id: string, enabled: boolean) => void
  remove:     (id: string) => void
}

export const useSources = create<SourcesStore>()(
  persist(
    (set, get) => ({
      sources: {},
      ensureBundled: () => {
        const cur = get().sources
        let dirty = false
        const next = { ...cur }
        for (const m of bundledManifests) {
          if (!next[m.id]) {
            next[m.id] = {
              manifest:    m,
              origin:      'bundled',
              installedAt: Date.now(),
              enabled:     true,
            }
            dirty = true
          } else {
            // Bundled manifests are code, not user state. Always
            // refresh from the bundle so schema/selector changes ship
            // without requiring a version bump in every JSON file or
            // a localStorage migration. User-controlled fields
            // (origin, enabled) are preserved.
            const prev = next[m.id]!
            if (prev.manifest !== m) {
              next[m.id] = { ...prev, manifest: m }
              dirty = true
            }
          }
        }
        if (dirty) set({ sources: next })
      },
      setEnabled: (id, enabled) => {
        const s = get().sources[id]
        if (!s) return
        set({ sources: { ...get().sources, [id]: { ...s, enabled } } })
      },
      remove: (id) => {
        const s = get().sources[id]
        // Bundled sources can be disabled but not removed.
        if (!s || s.origin === 'bundled') return
        const next = { ...get().sources }
        delete next[id]
        set({ sources: next })
      },
    }),
    {
      name:    'typoon.sources.v7',
      storage: createJSONStorage(() => localStorage),
    },
  ),
)

// ── selectors ──────────────────────────────────────────────────────

// Derived list — `Object.values().filter()` produces a brand-new
// array on every call. Without `useShallow`, zustand's
// `useSyncExternalStore` compares the snapshot by reference, sees a
// "new" value on every render, and re-renders the consumer — which
// can deadlock parents that `setState` in an effect off the list
// (Maximum update depth exceeded). `useShallow` compares array
// contents instead, so equivalent lists short-circuit re-render.
export function useEnabledSources(): InstalledSource[] {
  return useSources(
    useShallow((s) =>
      Object.values(s.sources).filter((x) => x.enabled),
    ),
  )
}

export function getSource(id: string): InstalledSource | null {
  return useSources.getState().sources[id] ?? null
}
