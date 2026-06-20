// Reader — /r/$workId/$numberNorm.

import { createFileRoute } from '@tanstack/react-router'

import { ReaderCacheProvider } from '@features/reader/cache/ReaderCacheProvider'
import { ReaderProvider } from '@features/reader/ReaderContext'
import { ReaderShell } from '@features/reader/components/ReaderShell'
import { WorkPageProvider } from '@features/work/contexts/WorkPageProvider'
import { useWork } from '@features/works/queries'
import { Spinner } from '@shared/ui/primitives'
import { EmptyState } from '@shared/ui/EmptyState'


function ReaderPage() {
  const { workId, numberNorm } = Route.useParams()
  const work = useWork(workId)

  if (work.isPending) return <div className="min-h-screen flex items-center justify-center bg-bg"><Spinner size={20} /></div>
  if (!work.data)      return <div className="min-h-screen flex items-center justify-center bg-bg p-6"><EmptyState title="Work không tồn tại" /></div>

  return (
    <WorkPageProvider work={work.data} workId={workId}>
      <ReaderCacheProvider>
        <ReaderProvider workId={workId} chapterRef={numberNorm}>
          <ReaderShell />
        </ReaderProvider>
      </ReaderCacheProvider>
    </WorkPageProvider>
  )
}


export const Route = createFileRoute('/r/$workId/$numberNorm')({
  component: ReaderPage,
  staticData: { auth: 'required', chrome: 'bare' },
})
