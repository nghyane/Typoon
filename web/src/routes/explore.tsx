// /explore — discover manga across all installed sources.
//
// Slice 1 stub. Slice 6 will add: global search, per-source carousels
// (popular / latest), and a "Cộng đồng đang dịch" section pulling from
// cross-guild draft activity.

import { createFileRoute } from '@tanstack/react-router'
import { Compass } from 'lucide-react'
import { EmptyState } from '@shared/ui/EmptyState'

function ExplorePage() {
  return (
    <div className="px-4 sm:px-6 py-4 sm:py-6">
      <div className="flex items-center justify-between mb-4 gap-3">
        <h1 className="text-lg font-semibold text-text tracking-tight">
          Khám phá
        </h1>
      </div>

      <div className="py-12 max-w-md mx-auto">
        <EmptyState
          icon={Compass}
          title="Đang xây dựng"
          hint="Khám phá manga theo nguồn, tìm kiếm cross-source, và xem manga cộng đồng đang dịch nhiều — sắp ra mắt."
        />
      </div>
    </div>
  )
}

export const Route = createFileRoute('/explore')({
  component: ExplorePage,
})
