// SourcePicker — pick the raw reading source for the current chapter.
//
//   ┌────────────────────────────────────────────────┐
//   │ VI                                             │
//   │   VI · TruyenQQ                       2h trước │
//   │                                                │
//   │ Khác                                           │
//   │   EN · Mangadex                            5h   │
//   │   JA · Niadd                              3d   │
//   └────────────────────────────────────────────────┘
//
// Translation runs inside the reader, not from this picker.

import { useMemo } from 'react'

import { BottomSheet } from '@shared/ui/BottomSheet'
import { Popover } from '@shared/ui/Popover'
import { cn } from '@shared/lib/cn'
import { timeAgo } from '@shared/lib/time'
import { useIsDesktop } from '@shared/lib/useMediaQuery'

import { useWorkChapters } from '@features/work/contexts/WorkChaptersContext'
import { useWorkIdentity } from '@features/work/contexts/WorkIdentityContext'
import type { SourceVersion } from '@features/work/data/types'

import { useReader } from '../ReaderContext'
import { useSourcePref, useSetSourcePref } from '../hooks/useSourcePref'
import { versionKeyOf } from '../data/selectors/resolveSource'


interface Props {
  open:    boolean
  onClose: () => void
  anchorRef: React.RefObject<HTMLButtonElement | null>
}


export function SourcePicker({ open, onClose, anchorRef }: Props) {
  const isDesktop = useIsDesktop()

  if (isDesktop) {
    return (
      <Popover
        open={open}
        onClose={onClose}
        anchorRef={anchorRef}
        align="end"
        minWidth={240}
        maxWidth={280}
      >
        <Body onClose={onClose} />
      </Popover>
    )
  }
  return (
    <BottomSheet open={open} onClose={onClose} title="Nguồn">
      <Body onClose={onClose} />
    </BottomSheet>
  )
}


// ── Body ───────────────────────────────────────────────────────


function Body({ onClose }: { onClose: () => void }) {
  const { workId, chapterRef } = useReader()
  const { work }   = useWorkIdentity()
  const { merged } = useWorkChapters()
  const pref       = useSourcePref(workId)
  const setPref    = useSetSourcePref(workId)

  const tgt = work.target_lang.toLowerCase()
  const chapter = merged.find(c => c.numberNorm === chapterRef) ?? null
  const versions = chapter?.sourceVersions ?? []

  const sortedVersions = useMemo(() => {
    const sortByDate = (a: SourceVersion, b: SourceVersion) => {
      const ad = a.ref.date ?? ''
      const bd = b.ref.date ?? ''
      return bd.localeCompare(ad)
    }
    return [...versions].sort((a, b) => {
      if (a.lang === tgt && b.lang !== tgt) return -1
      if (a.lang !== tgt && b.lang === tgt) return 1
      return sortByDate(a, b)
    })
  }, [versions, tgt])

  // Which row reads as "active right now"?
  //   - pref.raw + matching version → that row
  //   - pref.auto → first target-lang raw, else first other-lang raw
  const activeKey: string = useMemo(() => {
    if (pref.kind === 'raw')        return pref.versionKey
    // auto:
    if (sortedVersions[0]) return versionKeyOf(sortedVersions[0])
    return ''
  }, [pref, sortedVersions])

  function pickRaw(versionKey: string) {
    setPref({ kind: 'raw', versionKey })
    onClose()
  }

  if (sortedVersions.length === 0) {
    return (
      <p className="px-4 py-6 text-sm text-text-subtle text-center">
        Chưa có nguồn.
      </p>
    )
  }

  return (
    <div className="flex flex-col max-h-[70dvh] sm:max-h-[70vh]">
      <div className="flex-1 overflow-y-auto p-2">
        {sortedVersions.map(v => {
          const vk = versionKeyOf(v)
          return (
            <RawRow
              key={vk}
              version={v}
              active={activeKey === vk}
              onPick={() => pickRaw(vk)}
            />
          )
        })}
      </div>
    </div>
  )
}


// ── Raw row ────────────────────────────────────────────────────


function RawRow({
  version: v, active, onPick,
}: {
  version:     SourceVersion
  active:      boolean
  onPick:      () => void
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      aria-pressed={active}
      className={cn(
        'grid w-full grid-cols-[1rem_2rem_minmax(0,1fr)] items-center gap-2 h-8 px-2 rounded-sm text-left',
        'transition-colors duration-150 cursor-pointer',
        active
          ? 'text-text'
          : 'hover:bg-hover',
      )}
    >
      <span className="text-accent text-sm leading-none">{active ? '✓' : ''}</span>
      <span className={cn(
        'text-xs uppercase font-semibold tabular-nums',
        active ? 'text-accent' : 'text-text-subtle',
      )}>
        {v.lang.toUpperCase()}
      </span>
      <span className="min-w-0 truncate text-sm text-text">
        {v.source.manifest.name}
        {v.ref.scanlator && (
          <span className="text-text-subtle"> · @{v.ref.scanlator}</span>
        )}
        {v.ref.date && !v.ref.scanlator && (
          <span className="text-text-subtle"> · {timeAgo(v.ref.date)}</span>
        )}
      </span>
    </button>
  )
}
