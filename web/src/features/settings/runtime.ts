import { useQuery } from '@tanstack/react-query'
import { api, type PublicSettings } from '@shared/api/api'
import { qk } from '@shared/api/keys'

export type { PublicSettings }

export function usePublicSettings() {
  return useQuery<PublicSettings>({
    queryKey: qk.publicSettings(),
    queryFn:  api.getSettings,
    staleTime: 5 * 60_000,
    gcTime:    30 * 60_000,
    refetchOnWindowFocus: false,
  })
}
