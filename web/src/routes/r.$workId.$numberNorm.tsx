// Reader — /r/$workId/$numberNorm.
//
// Pipeline:
//   useSourcePref(workId)
//     → sticky preference (auto / translated / raw versionKey)
//
//   useChapterSources(workId, chapterRef)
//     → saved + job + raw versions (live IDB queries)
//
//   useActiveSource(workId, chapterRef)
//     → resolveSource(pref, sources) → ActiveSource (+ lazy raw probe)
//
//   useReaderSource(active, key)
//     → cache pool resolves source, retain/release lifecycle
//
//   ReaderShell renders, child views consume `source` + `sourceKey`.

import { createFileRoute } from '@tanstack/react-router'

import { useActiveSource } from '@features/reader/data/queries/useActiveSource'
import { useReaderSource } from '@features/reader/data/queries/useReaderSource'
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

  if (work.isPending) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg">
        <Spinner size={20} />
      </div>
    )
  }
  if (!work.data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg p-6">
        <EmptyState title="Work không tồn tại" />
      </div>
    )
  }

  return (
    <WorkPageProvider work={work.data} workId={workId}>
      <ReaderCacheProvider>
        <ReaderRoot
          key={`${workId}:${numberNorm}`}
          workId={workId}
          chapterRef={numberNorm}
        />
      </ReaderCacheProvider>
    </WorkPageProvider>
  )
}


function ReaderRoot({
  workId, chapterRef,
}: {
  workId:     string
  chapterRef: string
}) {
  return (
    <ReaderProvider workId={workId} chapterRef={chapterRef}>
      <ReaderBody workId={workId} chapterRef={chapterRef} />
    </ReaderProvider>
  )
}


function ReaderBody({
  workId, chapterRef,
}: {
  workId:     string
  chapterRef: string
}) {
  const { active, key, loading } = useActiveSource(workId, chapterRef)
  const reader = useReaderSource(active, key)

  const isLoading = reader.status === 'loading' || reader.status === 'idle' ||
    (reader.status === 'no-source' && loading)

  if (isLoading) {
    return <Placeholder />
  }
  if (reader.status === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg p-6">
        <EmptyState title="Không đọc được chương" hint={reader.error.message} />
      </div>
    )
  }
  if (reader.status === 'no-source') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-bg p-6">
        <EmptyState
          title="Chưa có nguồn để đọc"
          hint="Mở trang truyện và dịch chương này, hoặc lưu offline trước."
        />
      </div>
    )
  }

  return <ReaderShell source={reader.source} sourceKey={key} />
}


function Placeholder() {
  return (
    <div className="fixed inset-0 bg-bg flex items-center justify-center">
      <Spinner size={24} />
    </div>
  )
}


export const Route = createFileRoute('/r/$workId/$numberNorm')({
  component:  ReaderPage,
  staticData: { auth: 'required', chrome: 'bare' },
})
