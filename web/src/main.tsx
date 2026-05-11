import './index.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider, createRouter } from '@tanstack/react-router'
import { QueryClient, QueryClientProvider, keepPreviousData } from '@tanstack/react-query'
import { routeTree } from './routeTree.gen'

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

const router = createRouter({
  routeTree,
  // Cross-fade pending → resolved instead of blanking the route.
  defaultPendingMs: 200,
  defaultPendingMinMs: 0,
})

declare module '@tanstack/react-router' {
  interface Register { router: typeof router }
  // Per-route shell + auth metadata. Read by AppLayout via useMatches().
  // Defaults: chrome='app', auth='required'.
  interface StaticDataRouteOption {
    chrome?: 'app' | 'bare'
    auth?:   'public' | 'required'
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)
