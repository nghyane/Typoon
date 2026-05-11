import { createFileRoute, Outlet, Link } from '@tanstack/react-router'
import { useEffect } from 'react'
import { Compass } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { useSources } from '@features/browse/sources'

// Layout for `/browse/$source/*` — resolves the source manifest from
// the registry and provides a 404 surface when the user lands on a
// removed/disabled source.
function SourceLayout() {
  const { source: id } = Route.useParams()
  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const source = useSources((s) => s.sources[id])
  if (!source || !source.enabled) {
    return (
      <div className="px-6 py-16">
        <EmptyState
          icon={Compass}
          title="Không tìm thấy nguồn"
          hint="Nguồn này chưa được cài hoặc đã bị tắt."
          action={<Link to="/browse"><Button>Về trang nguồn</Button></Link>}
        />
      </div>
    )
  }

  return <Outlet />
}

export const Route = createFileRoute('/browse/$source')({
  component: SourceLayout,
  // Pre-resolve so children see a hydrated registry on first paint.
  loader: () => { useSources.getState().ensureBundled() },
})
