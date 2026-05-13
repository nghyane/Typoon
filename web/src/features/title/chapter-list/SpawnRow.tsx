// SpawnRow — `ChapterRow` wrapper that owns the per-row spawn hook.
//
// Library surface uses this so each row has its own progress state.
// Public-read surfaces render `<ChapterRow spawn={null} />` directly.

import { useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { api } from '@shared/api/api'
import type { HubChapter } from '../mergeChapters'
import { useSpawnChapter } from '../useSpawnChapter'
import { ChapterRow } from './ChapterRow'
import type { RowActions, SelectionState } from './types'

interface Props {
  chapter:       HubChapter
  targetLang:    string | null
  materialTitle: string
  selection:     SelectionState | null
}

export function SpawnRow({
  chapter, targetLang, materialTitle, selection,
}: Props) {
  const nav = useNavigate()
  const qc  = useQueryClient()
  const { progress, spawn, reset } = useSpawnChapter(targetLang ?? '')

  const actions: RowActions = {
    onSpawn: (ctx) => {
      if (!ctx.spawnableRaw) return
      spawn(ctx.spawnableRaw, chapter.label)
    },
    onOpenRaw: (ctx) => {
      if (!ctx.anyRaw?.upstreamUrl || !ctx.anyRaw.sourceId) return
      nav({
        to: '/raw',
        search: {
          source:     ctx.anyRaw.sourceId,
          url:        ctx.anyRaw.upstreamUrl,
          title:      materialTitle,
          number:     chapter.number,
          label:      chapter.label ?? undefined,
          materialId: ctx.anyRaw.materialId,
          numberNorm: ctx.anyRaw.numberNorm ?? undefined,
        },
      })
    },
    onRedo: async (ctx) => {
      if (!ctx.doneTranslation?.translationId) return
      await api.redoTranslation(ctx.doneTranslation.translationId)
      await qc.invalidateQueries({
        queryKey: ['material', 'detail', ctx.doneTranslation.materialId],
      })
    },
    onDelete: async (ctx) => {
      if (!ctx.doneTranslation?.translationId) return
      if (!confirm('Xóa bản dịch này?')) return
      await api.deleteTranslation(ctx.doneTranslation.translationId)
      await qc.invalidateQueries({
        queryKey: ['material', 'detail', ctx.doneTranslation.materialId],
      })
    },
  }

  return (
    <ChapterRow
      chapter={chapter}
      targetLang={targetLang}
      materialTitle={materialTitle}
      actions={actions}
      selection={selection}
      spawn={progress}
      spawnReset={reset}
    />
  )
}
