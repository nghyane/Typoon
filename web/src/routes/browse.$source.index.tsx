import { createFileRoute } from '@tanstack/react-router'
import { BrowseSourceHome } from '@features/browse/views/BrowseSourceHome'
import { useResolvedSource } from '@features/browse/useResolvedSource'

function SourceIndexPage() {
  const source = useResolvedSource()
  if (!source) return null
  return <BrowseSourceHome source={source} />
}

export const Route = createFileRoute('/browse/$source/')({
  component: SourceIndexPage,
})
