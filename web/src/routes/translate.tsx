import { createFileRoute, Outlet } from '@tanstack/react-router'

export const Route = createFileRoute('/translate')({
  component: () => <Outlet />,
})
