// /w/$workId — Work detail page.
//
// Route is a thin gate: handle loading / not-found / auth before the
// page components mount. Once `work.data` exists, render the page
// inside `WorkPageProvider` so descendant components consume the same
// data composition (identity / chapters / actions) via context — no
// per-component query orchestration.

import { useEffect } from 'react'
import { createFileRoute } from '@tanstack/react-router'

import { useWork, useTouchWork } from '@features/works/queries'
import { WorkPageProvider } from '@features/work/contexts/WorkPageProvider'
import { WorkPageContent } from '@features/work/components/WorkPageContent'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'
import { useHeaderStore } from '../store/header'


function WorkPage() {
  const { workId }  = Route.useParams()
  const work        = useWork(workId)
  const touch       = useTouchWork()
  const setHeader   = useHeaderStore(s => s.set)
  const clearHeader = useHeaderStore(s => s.clear)

  // Touch on mount → drives "Recently opened" rail.
  useEffect(() => {
    if (work.data) void touch(work.data.id)
  }, [work.data?.id, touch])

  // Header: back button to library, no title (h1 lives in the hero).
  useEffect(() => {
    setHeader('', [{ label: 'Thư viện', to: '/library' }])
    return () => clearHeader()
  }, [setHeader, clearHeader])

  if (work.isPending) {
    return (
      <div className="flex items-center justify-center py-16">
        <Spinner size={20} />
      </div>
    )
  }
  if (!work.data) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16">
        <EmptyState
          title="Work không tồn tại"
          hint="ID này không có hoặc đã bị xoá."
        />
      </div>
    )
  }

  return (
    <WorkPageProvider work={work.data} workId={workId}>
      <WorkPageContent />
    </WorkPageProvider>
  )
}


export const Route = createFileRoute('/w/$workId')({
  component:  WorkPage,
  staticData: { auth: 'required' },
})
