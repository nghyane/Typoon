// useSourcePref — sticky per-work source preference.
//
// Reads from the settings blob's `reader_source_prefs` map. Writes
// merge into the existing map. Default = { kind: 'auto' }.

import { useCallback, useMemo } from 'react'

import {
  useLocalSettings, useUpdateLocalSettings,
} from '@features/settings/local'
import {
  DEFAULT_PREF, type SourcePref,
} from '../data/types'


export function useSourcePref(workId: string): SourcePref {
  const q = useLocalSettings()
  return useMemo<SourcePref>(() => {
    const m = q.data?.reader_source_prefs ?? {}
    const pref = m[workId]
    if (!pref) return DEFAULT_PREF
    if (pref.kind === 'auto') return pref
    if (pref.kind === 'raw')  return pref
    return DEFAULT_PREF
  }, [q.data, workId])
}


export function useSetSourcePref(workId: string) {
  const update = useUpdateLocalSettings()
  const q      = useLocalSettings()
  return useCallback((pref: SourcePref) => {
    const prev = q.data?.reader_source_prefs ?? {}
    update.mutate({
      reader_source_prefs: { ...prev, [workId]: pref },
    })
  }, [workId, update, q.data])
}
