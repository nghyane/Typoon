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
//   • workers / quota / me / tokens     — realtime or auth-sensitive.
//   • library                            — short-lived, refetch-fast.
//   • community / recent-reads / search — user-action-driven, ephemeral.
//
// Storage: IndexedDB via `idb-keyval` (one named store, single key for
// the dehydrated cache blob). Survives 5+ MB without quota grief.

import { get, set, del, createStore } from 'idb-keyval'
import { QueryClient } from '@tanstack/react-query'
import { createAsyncStoragePersister } from '@tanstack/query-async-storage-persister'
import { persistQueryClient } from '@tanstack/react-query-persist-client'


/** Domain prefixes whose entries are safe to persist. Matched against
 *  `query.queryKey[0]` so adding a new top-level domain requires an
 *  explicit opt-in here. */
const PERSIST_DOMAINS = new Set<string>([
  'manifest',
  'translation',
  'work',
])


/** Bump when the persisted-cache shape changes incompatibly. The
 *  persister wipes old blobs that don't match. */
const CACHE_BUSTER = 'v1'

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


/** Wire persistence onto a QueryClient. Call once during app boot.
 *  Returns a teardown for hot-reload safety, not used in prod. */
export function installPersistence(qc: QueryClient): () => void {
  const persister = createAsyncStoragePersister({
    storage:      idbStorage,
    // Throttle write-back so we don't hammer IDB on every query
    // resolve. 1s is fast enough that a navigation away preserves
    // the latest data without serialising on every fetch.
    throttleTime: 1000,
  })

  const [unsubscribe] = persistQueryClient({
    queryClient: qc,
    persister,
    // Drop persisted entries older than 24h. Manifest data goes
    // stale fast enough that a day-old snapshot is rarely useful.
    maxAge:      24 * 60 * 60 * 1000,
    buster:      CACHE_BUSTER,
    dehydrateOptions: {
      shouldDehydrateQuery: (q) => {
        // Persist only on success — failed queries write nothing,
        // so a transient error doesn't pollute the next session.
        if (q.state.status !== 'success') return false
        const head = q.queryKey[0]
        return typeof head === 'string' && PERSIST_DOMAINS.has(head)
      },
    },
  })

  return unsubscribe
}


/** Imperative clear — for the Settings → "Xoá cache" affordance. */
export async function clearPersistedCache(): Promise<void> {
  await del(STORAGE_KEY, STORE)
}
