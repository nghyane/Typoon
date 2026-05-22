// Quota — server-side counter snapshot.

import { useQuery } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import { qk } from '@shared/api/keys'

export function useQuota() {
  return useQuery({
    queryKey:  qk.quota(),
    queryFn:   () => api.quota(),
    staleTime: 60_000,
  })
}
