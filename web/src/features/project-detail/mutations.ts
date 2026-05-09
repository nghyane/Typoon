import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@shared/api/api'
import { toast } from '@shared/ui/Toaster'

// =============================================================================
// Chapter mutations — hoisted out of <ChapterRow/> so mounting 100 rows does
// not create 200 mutation instances. Components call `redo.mutate(chapter_id)`.
// =============================================================================

export function useChapterMutations(projectId: number) {
  const qc = useQueryClient()

  // After every mutation we refresh both the chapter list (so the row
  // updates immediately) and the global workers indicator in the
  // header (so the user sees their action reflected in the queue
  // count without waiting for the next poll tick).
  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ['projects', projectId, 'chapters'] })
    qc.invalidateQueries({ queryKey: ['workers'] })
  }

  const redo = useMutation({
    mutationFn: (chapterId: number) => api.redoChapter(projectId, chapterId),
    onSuccess:  invalidate,
    onError:    (e: Error) => toast.error(e.message),
  })

  const remove = useMutation({
    mutationFn: ({ chapterId }: { chapterId: number; idx: number }) =>
      api.deleteChapter(projectId, chapterId),
    onSuccess: (_, { idx }) => {
      invalidate()
      toast.success(`Đã xoá chương ${idx}`)
    },
    onError: (e: Error) => toast.error(e.message),
  })

  return { redo, remove }
}
