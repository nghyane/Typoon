import './index.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import { QueryClient, keepPreviousData } from '@tanstack/react-query'
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client'
import { routeTree } from './routeTree.gen'
import {
  CACHE_BUSTER, MAX_AGE_MS, persister, shouldDehydrateQuery,
} from '@shared/api/persistence'
import { handleUnauthorized } from '@features/auth/session'
import { SourceFetchProvider } from '@features/browse/SourceFetchProvider'

// Persist DA flag before router strips query params from the URL.
if (new URLSearchParams(window.location.search).get('frame_id') != null) {
  sessionStorage.setItem('discord_activity', '1')
}

// Tag the document root when running inside Discord Activity so CSS
// can switch safe-area sources: DA uses `--discord-safe-area-inset-*`
// (set by the SDK on the iframe), plain browsers use `env(safe-area-
// inset-*)`. Detection is host-based — `.discordsays.com` only ever
// serves the DA iframe, the SDK isn't required.
if (
  window.location.hostname.endsWith('.discordsays.com') ||
  sessionStorage.getItem('discord_activity') === '1'
) {
  document.documentElement.classList.add('da-host')
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Keep cached data fresh for 60s — navigating between pages within
      // this window won't refetch and won't show skeletons.
      staleTime: 60_000,
      // Server-side state survives 5 min in cache after last unmount.
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      // While a query is fetching with new args (search/filter change),
      // continue showing the previous data — eliminates blank flash.
      placeholderData: keepPreviousData,
      retry: 1,
    },
  },
})

// Centralised 401 handling. `@shared/api/api` only knows about token
// storage; it dispatches `typoon:unauthorized` on every 401 from any
// request. We listen here (boot has the QueryClient) and clear the
// session cache too, so a stale `['session']` entry doesn't keep
// rendering a logged-in shell while AppLayout navigates to /login.
//
// AppLayout has its own listener on the same event for the nav side;
// the two listeners are independent — one owns auth state, the
// other owns routing.
window.addEventListener('typoon:unauthorized', () => {
  handleUnauthorized(queryClient)
})

const router = createRouter({
  routeTree,
  // Cross-fade pending → resolved instead of blanking the route.
  defaultPendingMs: 200,
  defaultPendingMinMs: 0,
  // Expose the QueryClient on `route.context` so route-level
  // `beforeLoad` can pre-fetch and intercept typed errors (e.g.
  // `WorkRedirectedError`) before the component mounts. No
  // useEffect-based redirect in components.
  context: { queryClient },
})

declare module '@tanstack/react-router' {
  interface Register { router: typeof router }
  // Per-route shell + auth metadata. Read by AppLayout via useMatches().
  // Defaults: chrome='app', auth='required'.
  interface StaticDataRouteOption {
    chrome?: 'app' | 'admin' | 'bare'
    auth?:   'public' | 'required'
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {/* PersistQueryClientProvider waits for IDB hydration before
        rendering children, so the first useQuery call after boot
        sees the persisted cache instead of firing a redundant
        network fetch. Without this wrapper RQ would refetch even
        when IDB has fresh data — defeats the persistence layer. */}
    <PersistQueryClientProvider
      client={queryClient}
      persistOptions={{
        persister,
        maxAge:           MAX_AGE_MS,
        buster:           CACHE_BUSTER,
        dehydrateOptions: { shouldDehydrateQuery },
      }}
    >
      <SourceFetchProvider>
        <RouterProvider router={router} />
      </SourceFetchProvider>
    </PersistQueryClientProvider>
  </StrictMode>,
)
