// ReaderShell — composition root for the reader page.
//
// Renders the page body (Pager / Strip) full-screen, with the top
// bar / bottom pill / picker / settings sheet overlaid. Coordinates
// chrome state via ReaderContext.

import { useCallback, useEffect, useRef, useState } from 'react'

import { PagerView } from '../PagerView'
import { StripView } from '../StripView'
import { TapZones } from './TapZones'
import { ReaderTopBar } from './ReaderTopBar'
import { ReaderBottomPill } from './ReaderBottomPill'
import { ReaderSettingsSheet } from './ReaderSettingsSheet'
import { ChapterPicker } from './ChapterPicker'
import { SourcePicker } from './SourcePicker'

import { useReader } from '../ReaderContext'
import { useReaderSettings, styleToLayout } from '../settings'
import { usePreloadNext } from '../hooks/usePreloadNext'
import { useRecordReading } from '@features/library/history'
import type { ReaderSource } from '../sources'


interface Props {
  source:    ReaderSource
  sourceKey: string
}


export function ReaderShell({ source, sourceKey }: Props) {
  const { workId, chapterRef, page, setPage, progress } = useReader()
  const settings = useReaderSettings()
  const layout   = styleToLayout(settings.style)
  const record   = useRecordReading()

  // Background warm-up for the next chapter once the reader passes
  // the threshold. Fire-and-forget; never blocks the current view.
  usePreloadNext({ workId, chapterRef, progress })

  const [pickerOpen,   setPickerOpen]   = useState(false)
  const [sourceOpen,   setSourceOpen]   = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const chapterTriggerRef = useRef<HTMLButtonElement>(null)
  const sourceTriggerRef  = useRef<HTMLButtonElement>(null)

  // Persist reading position (debounced via record's mutation).
  const handleChangePage = useCallback((next: number) => {
    setPage(next)
    record.mutate({
      work_id:     workId,
      chapter_ref: chapterRef,
      page:        next,
      total_pages: source.pageCount,
    })
  }, [setPage, record, workId, chapterRef, source.pageCount])

  // Reset page + scroll on chapter change. The route uses a `key` on
  // ReaderBody to remount the data hooks, but the outer DOM may
  // persist across renders if TanStack Router reuses the same
  // component instance; explicit window scroll reset covers both
  // cases.
  useEffect(() => {
    setPage(0)
    if (typeof window !== 'undefined') {
      window.scrollTo({ top: 0, behavior: 'auto' })
    }
  }, [chapterRef, setPage])

  return (
    <div className="fixed inset-0 bg-bg overflow-hidden">
      {/* Body */}
      <div className="absolute inset-0">
        {layout.mode === 'strip' ? (
          <StripView
            source={source}
            sourceKey={sourceKey}
            pageIndex={page}
            onChangePage={handleChangePage}
          />
        ) : (
          <>
            <PagerView
              source={source}
              sourceKey={sourceKey}
              pageIndex={page}
              onChangePage={handleChangePage}
              direction={layout.direction === 'rtl' ? 'rtl' : 'ltr'}
            />
            <TapZones pageCount={source.pageCount} />
          </>
        )}
      </div>

      {/* Chrome */}
      <ReaderTopBar
        onOpenChapters={() => setPickerOpen(true)}
        onOpenSources={() => setSourceOpen(true)}
        totalPages={source.pageCount}
        chapterTriggerRef={chapterTriggerRef}
        sourceTriggerRef={sourceTriggerRef}
      />
      <ReaderBottomPill
        onOpenSettings={() => setSettingsOpen(true)}
        totalPages={source.pageCount}
      />

      {/* Overlays */}
      <ChapterPicker
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        anchorRef={chapterTriggerRef}
      />
      <SourcePicker
        open={sourceOpen}
        onClose={() => setSourceOpen(false)}
        anchorRef={sourceTriggerRef}
      />
      <ReaderSettingsSheet
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  )
}
