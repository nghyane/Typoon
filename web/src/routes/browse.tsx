import { createFileRoute } from '@tanstack/react-router'
import { useEffect } from 'react'
import { Compass } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'
import { useHeaderStore } from '../store/header'

// Placeholder. /browse is being demoted: M4 makes "Thêm manga"
// inline from /library, and source picker moves to settings. We
// keep the route empty until the new flow ships so nav links don't
// 404.
function BrowsePage() {
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Duyệt nguồn', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  return (
    <div className="px-4 sm:px-6 pt-12 sm:pt-16">
      <EmptyState
        icon={Compass}
        title="Đang xây lại"
        hint="Duyệt nguồn sẽ tích hợp vào luồng Thêm manga ở Thư viện."
      />
    </div>
  )
}

export const Route = createFileRoute('/browse')({
  component: BrowsePage,
})
