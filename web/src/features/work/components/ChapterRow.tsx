// ChapterRow — pure display, no IDB hooks.
//
// Reads ChapterState via the per-row selector hook (Map lookup, O(1)).
// Actions delegated to ChapterActions which reads useWorkActions().

import { Link } from '@tanstack/react-router'

import { useWorkIdentity } from '../contexts/WorkIdentityContext'
import { useChapterState } from '../hooks/useChapterState'
import type { Row } from './ChapterList'
import { ChapterActions } from './ChapterActions'

import { cn } from '@shared/lib/cn'
import { languageName } from '@shared/lib/lang'
import { timeAgo } from '@shared/lib/time'


export function ChapterRow({ row, targetLang }: {
  row:        Row
  targetLang: string
}) {
  const { work } = useWorkIdentity()
  const state    = useChapterState(row.chapter.numberNorm)

  const { chapter, version } = row
  const langOfRow = version?.lang ?? targetLang

  return (
    <Link
      to="/r/$workId/$numberNorm"
      params={{ workId: work.id, numberNorm: chapter.numberNorm }}
      className={cn(
        'flex items-center gap-3 px-3 py-2.5',
        'cursor-pointer group',
        'focus-visible:outline-none focus-visible:bg-hover',
      )}
    >
      <span className="tabular-nums font-medium text-text-muted group-hover:text-text shrink-0 transition-colors">
        {chapter.number || chapter.numberNorm || '?'}
      </span>

      <LangChip lang={langOfRow} />

      <RowLabel row={row} className="hidden sm:flex sm:flex-1 sm:min-w-0 group-hover:text-text" />

      <span className="ml-auto sm:ml-0 inline-flex items-center gap-3 shrink-0">
        <span
          className="hidden sm:inline whitespace-nowrap text-text-subtle tabular-nums shrink-0"
          title={version?.ref.date ?? undefined}
        >
          {version?.ref.date ? timeAgo(version.ref.date) : ''}
        </span>
        <ChapterActions
          chapterRef={chapter.numberNorm}
          chapterNumber={chapter.number || chapter.numberNorm}
          version={version}
          targetLang={targetLang}
          state={state}
        />
      </span>
    </Link>
  )
}


function LangChip({ lang }: { lang: string }) {
  return (
    <span
      className="shrink-0 text-xs uppercase tabular-nums font-medium text-text-subtle"
      title={languageName(lang)}
    >
      {lang}
    </span>
  )
}


function RowLabel({ row, className }: { row: Row; className?: string }) {
  const { chapter, version } = row
  if (!version) {
    return (
      <span className={cn('truncate text-text', className)}>
        <span className="text-text-muted">Tải lên</span>
        {chapter.label && (
          <span className="text-text-subtle"> · {chapter.label}</span>
        )}
      </span>
    )
  }
  const sourceName = version.source.manifest.name
  const scanlator  = version.ref.scanlator
  return (
    <span className={cn('truncate text-text', className)}>
      {scanlator ? (
        <>
          <span className="text-text">@{scanlator}</span>
          <span className="text-text-subtle"> · {sourceName}</span>
        </>
      ) : (
        <span className="text-text-muted">{sourceName}</span>
      )}
    </span>
  )
}
