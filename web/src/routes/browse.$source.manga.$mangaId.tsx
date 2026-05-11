import { createFileRoute, Outlet } from '@tanstack/react-router'

// Layout for `/browse/$source/manga/$mangaId/*`. The detail page lives
// at the bare `/browse/$source/manga/$mangaId/` (index file); reader
// nests one level deeper.
export const Route = createFileRoute('/browse/$source/manga/$mangaId')({
  component: () => <Outlet />,
})
