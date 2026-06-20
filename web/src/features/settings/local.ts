// Local settings — single-row Dexie blob.

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { db, type SettingsBlob } from '@shared/db'
import { qk } from '@shared/api/keys'

export type { SettingsBlob }

const DEFAULT: SettingsBlob = {
  key:                 'global',
  theme:               'system',
  reader_mode:         'webtoon',
  default_target_lang: 'vi',
  updated_at:          new Date(0).toISOString(),
}

export function useLocalSettings() {
  return useQuery<SettingsBlob>({
    queryKey: qk.localSettings(),
    queryFn:  async () => normalizeSettings(await db().settings.get('global')),
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
        ...normalizeSettings(cur), ...clean,
        updated_at: new Date().toISOString(),
      }
      await db().settings.put(next)
      return next
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.localSettings() }),
  })
}

function normalizeSettings(value: SettingsBlob | undefined): SettingsBlob {
  if (!value) return DEFAULT
  return {
    ...DEFAULT,
    ...value,
    reader_mode: value.reader_mode === 'pager' ? 'webtoon' : value.reader_mode,
    default_target_lang: value.default_target_lang ?? DEFAULT.default_target_lang,
  }
}
