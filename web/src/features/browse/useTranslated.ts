// React Query bindings for `translate.ts`. Cache forever — translation
// of a given (text, target) is deterministic, no value in refetch.

import { useQuery } from '@tanstack/react-query'
import { translate, translateBatch } from './translate'

/** Translate a single string. Returns the translated value or `null`
 *  while loading / on failure. Caller falls back to the original. */
export function useTranslated(
  text: string | null | undefined,
  target: string,
  enabled: boolean = true,
): string | null {
  const t = text?.trim() ?? ''
  const { data } = useQuery({
    queryKey:  ['translate', t, target],
    queryFn:   () => translate(t, target),
    enabled:   enabled && t.length > 0,
    staleTime: Infinity,
    gcTime:    60 * 60_000,
    retry:     1,
  })
  return data ?? null
}

/** Batch variant. Returns an array of same length as `texts`; entries
 *  may be `null` while loading. Cache key is the joined text so
 *  re-orders re-fetch — keep input order stable. */
export function useTranslatedBatch(
  texts: string[],
  target: string,
  enabled: boolean = true,
): (string | null)[] {
  // Stable key from texts. The whole batch is one cache entry —
  // adding/removing chapters re-fetches, which is fine on first load
  // and rare otherwise.
  const key = texts.join('\u0000')
  const { data } = useQuery({
    queryKey:  ['translateBatch', key, target],
    queryFn:   () => translateBatch(texts, target),
    enabled:   enabled && texts.length > 0,
    staleTime: Infinity,
    gcTime:    60 * 60_000,
    retry:     1,
  })
  return data ?? texts.map(() => null)
}
