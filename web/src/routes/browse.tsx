import { createFileRoute, Outlet } from '@tanstack/react-router'

// Pass-through layout for `/browse/*`. The hub itself renders at the
// bare `/browse` (browse.index.tsx); deeper routes nest under here.
export const Route = createFileRoute('/browse')({
  component: () => <Outlet />,
})
