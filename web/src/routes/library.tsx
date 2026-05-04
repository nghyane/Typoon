import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/library')({
  component: () => <div className="p-8 text-zinc-500 text-sm">Library</div>,
})
