import { createFileRoute } from '@tanstack/react-router'
import { useEffect } from 'react'
import { Sparkles } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'
import { useHeaderStore } from '../store/header'

// Placeholder. M5 rebuilds /translate as the per-material memory
// editor (characters / world / style / glossary tabs). The old
// inbox lives inside /library cards via FollowButton + chapter row.
function TranslatePage() {
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Bản dịch', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  return (
    <div className="px-4 sm:px-6 pt-12 sm:pt-16">
      <EmptyState
        icon={Sparkles}
        title="Đang xây lại"
        hint="Trang quản lý bản dịch đang được rebuild — sẽ là editor ngữ cảnh dịch (translator memory) per-material."
      />
    </div>
  )
}

export const Route = createFileRoute('/translate')({
  component: TranslatePage,
})
