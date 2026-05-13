/** useMe — cached /api/me query.
 *
 *  Returns slim identity (id, display_name, avatar_url). Schema 19
 *  dropped the guild list from `/me`; the community is a single
 *  global pool now.
 */

import { useQuery } from '@tanstack/react-query'
import { api, type ApiMe } from '@shared/api/api'
import { qk } from '@shared/api/keys'

export function useMe(): {
  data:    ApiMe | null
  loading: boolean
} {
  const { data, isPending } = useQuery({
    queryKey: qk.me.self(),
    queryFn:  api.me,
    staleTime: 5 * 60_000,
  })
  return { data: data ?? null, loading: isPending }
}
