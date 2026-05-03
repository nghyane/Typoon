import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/settings')({
  component: () => (
    <div className="p-8" style={{ color: 'var(--color-text-2)' }}>
      Settings — coming soon
    </div>
  ),
})
