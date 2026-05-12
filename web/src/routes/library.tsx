import { createFileRoute } from '@tanstack/react-router'
import { useEffect } from 'react'
import { Library } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'
import { useHeaderStore } from '../store/header'

// Placeholder. M4 rebuilds /library as the unified hub-card grid
// (status chips + "+ Thêm manga" entry point). Until then the route
// exists so Sidebar/BottomNav stays linkable.
function LibraryPage() {
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Thư viện', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  return (
    <div className="px-4 sm:px-6 pt-12 sm:pt-16">
      <EmptyState
        icon={Library}
        title="Đang xây lại"
        hint="Thư viện đang được rebuild theo kiến trúc material+memory. Quay lại sau."
      />
    </div>
  )
}

export const Route = createFileRoute('/library')({
  component: LibraryPage,
})
