import { createFileRoute } from '@tanstack/react-router'
import { BrowseHub } from '@features/browse/views/BrowseHub'

export const Route = createFileRoute('/browse/')({
  component: BrowseHub,
})
