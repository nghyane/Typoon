import { createRootRoute, Outlet, useRouterState } from '@tanstack/react-router'
import { AppLayout } from '../components/AppLayout'

function RootShell() {
  const { location } = useRouterState()
  // Auth-related pages render bare — they handle their own redirects and
  // must not be wrapped by AppLayout (AppLayout redirects unauthenticated
  // users to /login, which would create a loop on /login or /auth/callback).
  if (location.pathname === '/login' || location.pathname.startsWith('/auth/')) {
    return <Outlet />
  }
  return (
    <AppLayout>
      <Outlet />
    </AppLayout>
  )
}

export const Route = createRootRoute({
  component: RootShell,
})
