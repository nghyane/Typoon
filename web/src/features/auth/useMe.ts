/** useMe — cached /api/me query.
 *
 *  Returns identity + Discord guild memberships. Drives the spawn
 *  modal's visibility picker (current guild) and the Hội Mê Truyện
 *  feed guild selector.
 */

import { useQuery } from '@tanstack/react-query'
import { api, type ApiMe } from '@shared/api/api'

export function useMe(): {
  data:    ApiMe | null
  loading: boolean
} {
  const { data, isPending } = useQuery({
    queryKey: ['me'],
    queryFn:  api.me,
    staleTime: 5 * 60_000,
  })
  return { data: data ?? null, loading: isPending }
}

/** First guild from /api/me — best-effort default when the SDK hasn't
 *  surfaced the active activity instance's guild_id. */
export function useDefaultGuildId(): string | null {
  const { data } = useMe()
  return data?.guilds?.[0]?.id ?? null
}
