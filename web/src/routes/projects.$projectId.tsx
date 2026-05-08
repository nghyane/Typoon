import { createFileRoute, Outlet } from '@tanstack/react-router'

// Pass-through layout for `/projects/$projectId/*`. The actual project
// detail page lives in `projects.$projectId.index.tsx` (renders at
// the bare `/projects/:id`); the chapter reader lives in
// `projects.$projectId.chapters.$chapterId.tsx`.
//
// Without this layout file, TanStack Router file-based routing has no
// place to mount nested child routes: any URL deeper than `/projects/:id/`
// would still match the parent route component, which never renders an
// `<Outlet/>`.
export const Route = createFileRoute('/projects/$projectId')({
  component: () => <Outlet />,
})
