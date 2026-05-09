import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api, type ApiChapter } from '@shared/api/api'
import { toast } from '@shared/ui/Toaster'

// =============================================================================
// Chapter mutations — hoisted out of <ChapterRow/> so mounting 100 rows does
// not create 200 mutation instances. Components call `redo.mutate(chapterId)`
// and `remove.mutate(chapterId)` — the mutation reads the chapter row from
// cache to build display strings, so callers never need to bundle display
// fields with the request.
// =============================================================================

export function useChapterMutations(projectId: number) {
  const qc = useQueryClient()

  const chaptersKey = ['projects', projectId, 'chapters'] as const

  // After every mutation we refresh both the chapter list (so the row
  // updates immediately) and the global workers indicator in the
  // header (so the user sees their action reflected in the queue
  // count without waiting for the next poll tick).
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: chaptersKey })
    qc.invalidateQueries({ queryKey: ['workers'] })
  }

  // Resolve the chapter row from cache — the pre-mutation snapshot, so
  // `remove` can still print a label after the row is gone.
  const findChapter = (chapterId: number): ApiChapter | undefined =>
    qc.getQueryData<ApiChapter[]>(chaptersKey)?.find((c) => c.chapter_id === chapterId)

  const redo = useMutation({
    mutationFn: (chapterId: number) => api.redoChapter(projectId, chapterId),
    onSuccess:  invalidate,
    onError:    (e: Error) => toast.error(e.message),
  })

  const start = useMutation({
    mutationFn: (chapterId: number) => api.startChapter(projectId, chapterId),
    onSuccess: (_, chapterId) => {
      const ch = findChapter(chapterId)
      invalidate()
      toast.success(ch ? `Đã bắt đầu Ch.${ch.number}` : 'Đã bắt đầu chương')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const startMany = useMutation({
    mutationFn: (chapterIds: number[]) => api.startChapters(projectId, chapterIds),
    onSuccess: (res) => {
      invalidate()
      if (res.started === 0) {
        toast.info('Không có chương nào để bắt đầu (đã chạy hoặc hoàn thành).')
      } else if (res.started < res.total) {
        toast.success(`Đã bắt đầu ${res.started}/${res.total} chương — bỏ qua các chương đã chạy.`)
      } else {
        toast.success(`Đã bắt đầu ${res.started} chương`)
      }
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const remove = useMutation({
    mutationFn: (chapterId: number) => api.deleteChapter(projectId, chapterId),
    onSuccess: (_, chapterId) => {
      const ch = findChapter(chapterId)
      invalidate()
      toast.success(ch ? `Đã xoá Ch.${ch.number}` : 'Đã xoá chương')
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return { redo, start, startMany, remove }
}
