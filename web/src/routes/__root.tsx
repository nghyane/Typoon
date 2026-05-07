import { createRootRoute, Outlet, useRouterState } from '@tanstack/react-router'
import { AppLayout } from '../components/AppLayout'

function RootShell() {
  const { location } = useRouterState()
  // /login renders standalone — it must not be wrapped by AppLayout
  // because AppLayout redirects to /login when not authenticated, which
  // would create a loop.
  if (location.pathname === '/login') {
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
