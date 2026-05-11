// Auto-translate preference — global toggle, persisted.
//
// "enabled" applies across all sources. We skip translation
// automatically when the source's primary language already matches
// the user's target (OTruyen is VI → no-op), so a user reading from
// both vi and non-vi sources doesn't have to toggle per-source.

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

interface AutoTranslateStore {
  enabled: boolean
  target:  string             // BCP-47, default 'vi'
  setEnabled: (v: boolean) => void
  setTarget:  (t: string) => void
}

export const useAutoTranslate = create<AutoTranslateStore>()(
  persist(
    (set) => ({
      enabled:    true,
      target:     'vi',
      setEnabled: (v) => set({ enabled: v }),
      setTarget:  (t) => set({ target: t }),
    }),
    {
      name:    'typoon.autoTranslate.v1',
      storage: createJSONStorage(() => localStorage),
    },
  ),
)

/** Decide whether to translate text from a source whose chapters
 *  are in `sourceLangs`. Returns false when the source already
 *  speaks the target language (OTruyen vi → vi). */
export function shouldTranslate(
  enabled: boolean,
  target: string,
  sourceLangs: string[],
): boolean {
  if (!enabled) return false
  if (sourceLangs.includes(target)) return false
  return true
}
