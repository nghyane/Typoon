// Local settings — single-row Dexie blob.

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { db, type SettingsBlob } from '@shared/db'
import { qk } from '@shared/api/keys'

export type { SettingsBlob }

const DEFAULT: SettingsBlob = {
  key:                 'global',
  theme:               'system',
  reader_mode:         'pager',
  default_target_lang: null,
  updated_at:          new Date(0).toISOString(),
}

export function useLocalSettings() {
  return useQuery<SettingsBlob>({
    queryKey: qk.settings(),
    queryFn:  async () => (await db().settings.get('global')) ?? DEFAULT,
    staleTime: Infinity,
  })
}

export function useUpdateLocalSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (patch: Partial<Omit<SettingsBlob, 'key' | 'updated_at'>>) => {
      const cur = (await db().settings.get('global')) ?? DEFAULT
      // Drop undefined keys so a partial patch doesn't blank fields
      // the caller didn't mean to touch.
      const clean: Partial<SettingsBlob> = {}
      for (const [k, v] of Object.entries(patch)) {
        if (v !== undefined) (clean as Record<string, unknown>)[k] = v
      }
      const next: SettingsBlob = {
        ...cur, ...clean,
        updated_at: new Date().toISOString(),
      }
      await db().settings.put(next)
      return next
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.settings() }),
  })
}
