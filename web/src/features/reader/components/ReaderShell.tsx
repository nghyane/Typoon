// ReaderShell — composition root for the reader page.
//
// Keeps existing chrome (top bar, bottom pill, pickers) but replaces
// the data layer with new hooks: useChapterPages.

import { useEffect, useMemo, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'

import { PageRenderer } from '../PageRenderer'
import { ReaderTopBar } from './ReaderTopBar'
import { ReaderBottomPill } from './ReaderBottomPill'
import { ReaderSettingsSheet } from './ReaderSettingsSheet'
import { ChapterPicker } from './ChapterPicker'
import { SourcePicker } from './SourcePicker'
import { useChapterPages } from '../hooks/useChapterPages'

import { useReader } from '../ReaderContext'
import { usePreloadNext } from '../hooks/usePreloadNext'
import { useActiveSource } from '../data/queries/useActiveSource'


export function ReaderShell() {
  const { workId, chapterRef, setPage, progress } = useReader()

  usePreloadNext({ workId, chapterRef, progress })

  const { active, loading } = useActiveSource(workId, chapterRef)
  console.warn('[ReaderShell] active.kind=', active.kind, 'loading=', loading, 'urls=', active.kind === 'raw-online' ? (active as any).urls?.length : 0)
  const urls = active.kind === 'raw-online' ? active.urls : [] as readonly string[]
  const pageKey = useMemo(() => urls.join('\n'), [urls])
  const { blobs } = useChapterPages(urls, pageKey)

  const [pickerOpen,   setPickerOpen]   = useState(false)
  const [sourceOpen,   setSourceOpen]   = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const chapterTriggerRef = useRef<HTMLButtonElement>(null)
  const sourceTriggerRef  = useRef<HTMLButtonElement>(null)

  useEffect(() => { setPage(0); window.scrollTo(0, 0) }, [chapterRef, setPage])

  // Loading — active source not resolved yet.
  if (loading || active.kind === 'none') {
    return (
      <div className="fixed inset-0 bg-bg flex items-center justify-center">
        <Loader2 size={24} className="animate-spin" />
      </div>
    )
  }

  return (
    <div className="fixed inset-0 bg-bg overflow-hidden">
      {/* Body — canvas pages in scroll strip */}
      <div className="absolute inset-0 overflow-y-auto">
        <div className="flex flex-col items-center">
          {blobs.map((blob, i) => (
            <PageRenderer key={i} blob={blob} index={i} className="w-full max-w-3xl" />
          ))}
        </div>
      </div>

      {/* Chrome */}
      <ReaderTopBar
        onOpenChapters={() => setPickerOpen(true)}
        onOpenSources={() => setSourceOpen(true)}
        totalPages={urls.length}
        chapterTriggerRef={chapterTriggerRef}
        sourceTriggerRef={sourceTriggerRef}
      />
      <ReaderBottomPill totalPages={urls.length} onOpenSettings={() => setSettingsOpen(true)} />

      <ChapterPicker open={pickerOpen} onClose={() => setPickerOpen(false)} anchorRef={chapterTriggerRef} />
      <SourcePicker open={sourceOpen} onClose={() => setSourceOpen(false)} anchorRef={sourceTriggerRef} />
      <ReaderSettingsSheet open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  )
}
