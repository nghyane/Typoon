import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/settings')({
  component: () => <div className="p-8 text-zinc-500 text-sm">Settings</div>,
})
