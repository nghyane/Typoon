// React Query persistence — selective IndexedDB cache for the data
// types that benefit most from surviving reloads and source outages.
//
// What gets persisted (positive list, hand-picked):
//
//   • manifest.detail        — 3rd-party manga page. Source down →
//                              still browse what you've already opened.
//   • manifest.chapter-pages — chapter image URLs. Immutable once
//                              published; source down → still read.
//   • translation.byId       — archive_url is immutable once
//                              state='done'; cheap to keep so the
//                              reader opens instantly post-reload.
//   • work.byId              — short-TTL persist (60s effective) so
//                              navigation reload doesn't flash
//                              skeletons. Polling immediately reconciles.
//
// What's NOT persisted (negative list, explicit):
//
//   • workers / quota / session / tokens — realtime or auth-sensitive.
//   • library                            — short-lived, refetch-fast.
//   • community / recent-reads / search — user-action-driven, ephemeral.
//
// Storage: IndexedDB via `idb-keyval` (one named store, single key for
// the dehydrated cache blob). Survives 5+ MB without quota grief.
//
// Hydration is async by design (IDB is async). Consumers wrap their
// app in `PersistQueryClientProvider` from
// `@tanstack/react-query-persist-client` so React waits for hydration
// before rendering children — without that wrapper, the first render
// would fire fresh fetches even when IDB has data, defeating the
// point of persisting.

import { get, set, del, createStore } from 'idb-keyval'
import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister'
import type { PersistedClient, Persister } from '@tanstack/react-query-persist-client'


/** Domain prefixes whose entries are safe to persist. Matched against
 *  `query.queryKey[0]` so adding a new top-level domain requires an
 *  explicit opt-in here. */
const PERSIST_DOMAINS = new Set<string>([
  'manifest',   // source-adapter responses (manga catalog, immutable per request)
])


/** Bump when the persisted-cache shape changes incompatibly. The
 *  persister wipes old blobs that don't match. */
export const CACHE_BUSTER = 'v3.5'

/** Drop persisted entries older than this on hydrate. Manifest data
 *  goes stale fast enough that a day-old snapshot is rarely useful. */
export const MAX_AGE_MS = 24 * 60 * 60 * 1000

const STORE = createStore('typoon-rq', 'cache')
const STORAGE_KEY = 'rq-cache'


/** Adapter to TanStack's storage contract over `idb-keyval`. The
 *  persister hands us a single string-encoded blob; we shove it under
 *  one IDB key. The `key` parameter from the contract is ignored —
 *  we use a fixed key so the persister tracks one entry per app. */
const idbStorage = {
  getItem:    async (): Promise<string | null> =>
    (await get(STORAGE_KEY, STORE)) ?? null,
  setItem:    async (_: string, value: string): Promise<void> =>
    set(STORAGE_KEY, value, STORE),
  removeItem: async (): Promise<void> =>
    del(STORAGE_KEY, STORE),
}


/** Persister instance the `<PersistQueryClientProvider>` consumes. */
export const persister: Persister = createAsyncStoragePersister({
  storage:      idbStorage,
  // Throttle write-back so we don't hammer IDB on every query
  // resolve. 1s is fast enough that a navigation away preserves
  // the latest data without serialising on every fetch.
  throttleTime: 1000,
}) as Persister


/** Dehydrate filter — only persist queries on the positive list and
 *  only when they succeeded (failed queries write nothing, so a
 *  transient error doesn't pollute the next session). */
export function shouldDehydrateQuery(q: {
  state: { status: string }
  queryKey: readonly unknown[]
}): boolean {
  if (q.state.status !== 'success') return false
  const head = q.queryKey[0]
  return typeof head === 'string' && PERSIST_DOMAINS.has(head)
}


/** Imperative clear — for the Settings → "Xoá cache" affordance. */
export async function clearPersistedCache(): Promise<void> {
  await del(STORAGE_KEY, STORE)
}


/** Re-export the PersistedClient type so the wrapper at boot can
 *  reference it without dragging the whole persist-client module
 *  into main.tsx. */
export type { PersistedClient }
