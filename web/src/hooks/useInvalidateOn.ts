import { useQueryClient } from '@tanstack/react-query'
import { useSSE } from './useSSE'
import { chapterKeys, } from '../api/chapters'
import { projectKeys } from '../api/projects'
import { toast } from '../stores/toast'
import type { SSEEvent } from '../api/types'

export function useInvalidateOn() {
  const qc = useQueryClient()

  useSSE((event: SSEEvent) => {
    const { type, chapter_id } = event

    if (type === 'StageDone' || type === 'StageStarted') {
      qc.invalidateQueries({ queryKey: projectKeys.all() })
      if (chapter_id) {
        qc.invalidateQueries({ queryKey: chapterKeys.all(chapter_id) })
      }
    }

    if (type === 'PageDone' && chapter_id) {
      qc.invalidateQueries({ queryKey: chapterKeys.detail(
        event.project_id as number ?? 0, chapter_id
      )})
    }

    if (type === 'StageFailed') {
      const stage = event.stage ?? 'unknown'
      toast.error(`Chapter ${chapter_id}: ${stage} failed`)
      if (chapter_id) {
        qc.invalidateQueries({ queryKey: chapterKeys.all(chapter_id) })
      }
    }

    if (type === 'ChapterDownloaded') {
      toast.success(`Chapter downloaded`)
      qc.invalidateQueries({ queryKey: projectKeys.all() })
    }
  })
}
