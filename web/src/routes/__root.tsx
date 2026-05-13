import { createRootRouteWithContext, Outlet } from '@tanstack/react-router'
import type { QueryClient } from '@tanstack/react-query'
import { AppLayout } from '@app/AppLayout'

// Root always wraps in AppLayout; AppLayout decides shell + auth from
// each route's staticData. No pathname-based branching here.
//
// Context exposes the `QueryClient` so route-level `beforeLoad` can
// pre-fetch and intercept typed errors (e.g. `WorkRedirectedError`)
// without a useEffect-based redirect in the component layer.

export interface RouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: () => (
    <AppLayout>
      <Outlet />
    </AppLayout>
  ),
})
