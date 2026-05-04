import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/glossary')({
  component: () => <div className="p-8 text-zinc-500 text-sm">Glossary</div>,
})
