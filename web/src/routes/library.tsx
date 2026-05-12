import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { Plus, Library } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'
import { Button } from '@shared/ui/Button'
import { useHeaderStore } from '../store/header'
import { AddMangaModal } from '@features/library/addManga/AddMangaModal'
import { useSources } from '@features/browse/sources'

// =============================================================================
// /library — entry point.
//
// Slice 11 ships the Add-manga modal first; the grid + translation
// view land in slice 12. Until then the page is a single empty state
// with a primary CTA to open the modal, plus a "+" in the header
// once the user has at least one source installed.
// =============================================================================

function LibraryPage() {
  const setHeader   = useHeaderStore((s) => s.set)
  const clearHeader = useHeaderStore((s) => s.clear)
  useEffect(() => {
    setHeader('Thư viện', [])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  // Make sure the bundled manifests are hydrated before the user can
  // hit Add-manga — the fanout would otherwise see zero sources.
  const ensureBundled = useSources((s) => s.ensureBundled)
  useEffect(() => { ensureBundled() }, [ensureBundled])

  const [addOpen, setAddOpen] = useState(false)

  return (
    <div className="px-4 sm:px-6 pt-4 sm:pt-6">
      <div className="flex items-center justify-end mb-4">
        <Button variant="primary" onClick={() => setAddOpen(true)}>
          <Plus size={14} />
          Thêm manga
        </Button>
      </div>

      <div className="pt-12 sm:pt-16">
        <EmptyState
          icon={Library}
          title="Thư viện đang trống"
          hint="Dán đường dẫn manga hoặc gõ tên để thêm. Mỗi truyện bạn theo dõi sẽ về đây."
          action={
            <Button variant="primary" onClick={() => setAddOpen(true)}>
              <Plus size={14} />
              Thêm manga đầu tiên
            </Button>
          }
        />
      </div>

      <AddMangaModal open={addOpen} onClose={() => setAddOpen(false)} />
    </div>
  )
}

export const Route = createFileRoute('/library')({
  component: LibraryPage,
})
