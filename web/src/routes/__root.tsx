import { createRootRoute, Outlet } from '@tanstack/react-router'
import { AppLayout } from '@app/AppLayout'

// Root always wraps in AppLayout; AppLayout decides shell + auth from
// each route's staticData. No pathname-based branching here.
export const Route = createRootRoute({
  component: () => (
    <AppLayout>
      <Outlet />
    </AppLayout>
  ),
})
