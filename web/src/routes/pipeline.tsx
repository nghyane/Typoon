import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/pipeline')({
  component: () => (
    <div className="p-8" style={{ color: 'var(--color-text-2)' }}>
      Pipeline — coming soon
    </div>
  ),
})
